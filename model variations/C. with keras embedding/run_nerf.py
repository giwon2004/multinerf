import sys, os
sys.path.append(os.path.abspath('../..'))
from dataloader import *
from general_helper import *
from model import *

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time

tf.compat.v1.enable_eager_execution()


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, obj, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    obj = tf.broadcast_to(tf.constant(float(obj)), embedded.shape[:-1] + [1])
    embedded = tf.concat([embedded, obj], -1)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)
    
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays(ray_batch, obj, network_fn, network_query_fn, N_samples,
                retraw=False, lindisp=False, perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
	  obj: int. Contains the information of currently training obj.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.
        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-(act_fn(raw)+1e-10) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat([dists, tf.broadcast_to([1e10], dists[..., :1].shape)], axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map / tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    raw = network_query_fn(pts, viewdirs, obj, network_fn)  # [N_rays, N_samples, 4]
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, obj, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def batchify_rays(rays_flat, obj, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], obj, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, obj, 
           chunk=1024*32, rays=None, c2w=None, ndc=True, near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
	  obj: int. Contains the information of currently training obj.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near[obj] * tf.ones_like(rays_d[..., :1]), far[obj] * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, obj, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, obj, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, obj, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, input_ch_obj=1, 
        input_obj_dim=args.diversity, output_ch_obj=args.output_ch_obj, use_viewdirs=args.use_viewdirs)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, input_ch_obj=1, 
            input_obj_dim=args.diversity, output_ch_obj=args.output_ch_obj, use_viewdirs=args.use_viewdirs)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn(inputs, viewdirs, obj, network_fn): return run_network(
        inputs, viewdirs, obj, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


class ArgumentList:
    def __init__(self, argdict = {}):

        # training options
        self.netdepth = 8
            # layers in network
        self.netwidth = 256
            # channels per layer
        self.netdepth_fine = 8
            # layers in fine network
        self.netwidth_fine = 256
            # channels per layer in fine network
        self.N_rand = 32*32*4
            # batch size (number of random rays per gradient step)
        self.lrate = 5e-4
            # learning rate
        self.lrate_decay = 250
            # exponential learning rate decay (in 1000s)
        self.chunk = 1024*32
            # number of rays processed in parallel, decrease if running out of memory
        self.netchunk = 1024*64
            # number of pts sent through network in parallel, decrease if running out of memory
        self.no_batching = True
            # only take random rays from 1 image at a time
        self.no_reload = True
            # do not reload weights from saved ckpt
        self.ft_path = None
            # specific weights npy file to reload for coarse network
        self.random_seed = None
            # fix random seed for repeatability
        # pre-crop options
        self.precrop_iters = 0
            # number of steps to train on central crops
        self.precrop_frac = .5
            # fraction of img taken for central crops

        # rendering options
        self.N_samples = 64
            # number of coarse samples per ray
        self.N_importance = 0
            # number of additional fine samples per ray
        self.perturb = 1.
            # set to 0. for no jitter, 1. for jitter
        self.use_viewdirs = True
            # use full 5D input instead of 3D
        self.i_embed = 0
            # set 0 for default positional encoding, -1 for none
        self.multires = 10
            # log2 of max freq for positional encoding (3D location)
        self.multires_views = 4
            # log2 of max freq for positional encoding (2D direction)
        self.raw_noise_std = 0.
            # std dev of noise added to regularize sigma_a output, 1e0 recommended
        self.render_only = True
            # do not optimize, reload weights and render out render_poses path
        self.render_test = True
            # render the test set instead of render_poses path
        self.render_factor = 0
            # downsampling factor to speed up rendering, set 4 or 8 for fast preview

        # dataset options
        self.dataset_type = 'llff'
            # options: llff / blender / deepvoxels
        self.testskip = 8
            # will load 1/N images from test/val sets, useful for large datasets like deepvoxels

        # deepvoxels flags
        self.shape = 'greek'
            # options : armchair / cube / greek / vase

        # blender flags
        self.white_bkgd = True
            # set to render synthetic data on a white bkgd (always use for dvoxels)
        self.half_res = True
            # load blender synthetic data at 400x400 instead of 800x800
        
        # llff flags
        self.factor = 8
            # downsample factor for LLFF images
        self.no_ndc = True
            # do not use normalized device coordinates (set for non-forward facing scenes)
        self.lindisp = True
            # sampling linearly in disparity rather than depth
        self.spherify = True
            # set for spherical 360 scenes
        self.llffhold = True
            # will take every 1/N images as LLFF test set, paper uses 8

        # logging/saving options
        self.i_print = 100
            # frequency of console printout and metric loggin
        self.i_img = 500
            # frequency of tensorboard image logging
        self.i_weights = 10000
            # frequency of weight ckpt saving
        self.i_testset = 50000
            # frequency of testset saving
        self.i_video = 50000
            # frequency of render_poses video saving


        for key in argdict:
            assert type(key) == str
            if key in ["no_batching", "no_reload", "use_viewdirs", "render_only", 
                       "render_test", "white_bkgd", "half_res", "no_ndc", "lindisp", "spherify"]: 
                assert argdict[key] is True or argdict[key] is False 
            setattr(self, key, argdict[key])

def FETCH_ARGUMENT(argDict = {}):
    return ArgumentList(argDict)

def train():

    ARG_DICT = {"expname": "dvox_paper_cubegreek", 
                "shape": [("deepvoxels", "cube"), ("deepvoxels", "greek")], 
                "basedir": "/content/gdrive/MyDrive/NeRF/logs",
                "datadir": "/content/gdrive/MyDrive/NeRF/data",
                "dataset_type": "deepvoxels",
                "no_batching": True,
                "use_viewdirs": True,
                "white_bkgd": True,
                "no_reload": True, # originally True
                "lrate": 5e-4, # originally 5e-4
                "lrate_decay": 250,
                "N_samples": 64, # originally 64
                "N_importance": 64, # originally 128
                "N_rand": 1024, # originally 4096
                "render_only": False,
                "chunk": 1024 * 32, # originally default
                "netchunk": 1024 * 64, # originally default
                "i_img": 500, # originally default
                "i_print": 100, # originally default
                "i_weights": 1000, # originally default
                "output_ch_obj": 10,
                }
    
    args = FETCH_ARGUMENT(ARG_DICT)
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data

    args.diversity = len(args.shape)
    images = [None] * args.diversity
    poses = [None] * args.diversity
    render_poses = [None] * args.diversity
    hwf = [None] * args.diversity
    i_split = [None] * args.diversity
    i_train = [None] * args.diversity
    i_val = [None] * args.diversity
    i_test = [None] * args.diversity
    near = [None] * args.diversity
    far = [None] * args.diversity

    for i in range(args.diversity):
        if type(args.shape[i]) == str:
            dataset_type = args.shape[i]
        else:
            dataset_type = args.shape[i][0]
            shape = args.shape[i][1]

        if dataset_type == 'llff':
            images[i], poses[i], bds, render_poses[i], i_test[i] = load_llff_data(args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
            hwf[i] = poses[i][0, :3, -1]
            poses[i] = poses[i][:, :3, :4]
            print('Loaded llff', images[i].shape, render_poses[i].shape, hwf[i], args.datadir)
            if not isinstance(i_test, list):
                i_test[i] = [i_test]

            if args.llffhold > 0:
                print('Auto LLFF holdout,', args.llffhold)
                i_test[i] = np.arange(images[i].shape[0])[::args.llffhold]

            i_val[i] = i_test[i]
            i_train[i] = np.array([j for j in np.arange(int(images[i].shape[0])) if (j not in i_test[i] and j not in i_val[i])])

            print('DEFINING BOUNDS')
            if args.no_ndc:
                near[i] = tf.reduce_min(bds) * .9
                far[i] = tf.reduce_max(bds) * 1.
            else:
                near[i] = 0.
                far[i] = 1.
            print('NEAR FAR', near[i], far[i])

        elif dataset_type == 'blender':
            images[i], poses[i], render_poses[i], hwf[i], i_split[i] = load_blender_data(args.datadir, args.half_res, args.testskip)
            print('Loaded blender', images[i].shape, render_poses[i].shape, hwf[i], args.datadir)
            i_train[i], i_val[i], i_test[i] = i_split[i]

            near[i] = 2.
            far[i] = 6.

            if args.white_bkgd:
                images[i] = images[i][..., :3]*images[i][..., -1:] + (1.-images[i][..., -1:])
            else:
                images[i] = images[i][..., :3]

        elif dataset_type == 'deepvoxels':
 
            images[i], poses[i], render_poses[i], hwf[i], i_split[i] = load_dv_data(scene=shape, basedir=args.datadir, testskip=args.testskip)
            print('Loaded deepvoxels for', shape, images[i].shape, render_poses[i].shape, hwf[i], args.datadir[i])

            i_train[i], i_val[i], i_test[i] = i_split[i]

            hemi_R = np.mean(np.linalg.norm(poses[i][:, :3, -1], axis=-1))
            near[i] = hemi_R-1.
            far[i] = hemi_R+1.

        else:
            print('Unknown dataset type', args.dataset_type, 'excluding')

    # Cast intrinsics to right types
    for i in range(args.diversity):
        H, W, focal = hwf[i]
        H, W = int(H), int(W)
        hwf[i] = [H, W, focal]

    if args.render_test:
        for i in range(args.diversity):
            render_poses[i] = np.array(poses[i][i_test[i]])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(args)

    bds_dict = {'near': [], 'far': []}
    for i in range(args.diversity):
        bds_dict['near'].append(tf.cast(near[i], tf.float32))
        bds_dict['far'].append(tf.cast(far[i], tf.float32))

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        for i in range(args.diversity):
            if args.render_test:
                # render_test switches to test poses
                image = images[i][i_test[i]]
            else:
                # Default is smoother render_poses path
                image = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}_{}'.format('test' if args.render_test else 'path', start, i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            obj = i
            rgbs, _ = render_path(render_poses[i], hwf[i], obj, args.chunk, render_kwargs_test, gt_imgs=image, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    #global_step = tf.compat.v1.train.get_or_create_global_step()
    tf.summary.experimental.set_step(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        rays_rgb = [None] * args.diversity
        for i in range(args.diversity):
            H, W, focal = hwf[i]
            # For random ray batching.
            #
            # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
            # interpreted as,
            #   axis=0: ray origin in world space
            #   axis=1: ray direction in world space
            #   axis=2: observed RGB color of pixel
            print('get rays')
            # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
            # for each pixel in the image. This stack() adds a new dimension.
            rays = [get_rays_np(H, W, focal, p) for p in poses[i][:, :3, :4]]
            rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
            print('done, concats')
            # [N, ro+rd+rgb, H, W, 3]
            rays_rgb[i] = np.concatenate([rays, images[i][:, None, ...]], 1)
            # [N, H, W, ro+rd+rgb, 3]
            rays_rgb[i] = np.transpose(rays_rgb[i], [0, 2, 3, 1, 4])
            rays_rgb[i] = np.stack([rays_rgb[i][j] for j in i_train[i]], axis=0)  # train images only
            # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb[i] = np.reshape(rays_rgb[i], [-1, 3, 3])
            rays_rgb[i] = rays_rgb[i].astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_rgb[i])
            print('done')
            i_batch = 0

    N_iters = 1000000
    print('Begin')
    for i in range(args.diversity):
        print('TRAIN views are', i_train[i])
        print('TEST views are', i_test[i])
        print('VAL views are', i_val[i])

    # Summary writers
    writer = tf.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()
        loss = []; psnr = []; trans = []
        if args.N_importance > 0.: psnr0 = []

        for d in range(args.diversity):

            # Sample random ray batch
            H, W, focal = hwf[d]

            if use_batching:
                # Random over all images
                batch = rays_rgb[d][i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
                batch = tf.transpose(batch, [1, 0, 2])

                # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
                # target_s[n, rgb] = example_id, observed color.
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb[d].shape[0]:
                    np.random.shuffle(rays_rgb[d])
                    i_batch = 0

            else:
                # Random from one image
                img_i = np.random.choice(i_train[d])
                target = images[d][img_i]
                pose = poses[d][img_i, :3, :4]

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, focal, pose)
                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = tf.stack(tf.meshgrid(
                            tf.range(H//2 - dH, H//2 + dH), 
                            tf.range(W//2 - dW, W//2 + dW), 
                            indexing='ij'), -1)
                        if i < 10:
                            print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                    else:
                        coords = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'), -1)
                    coords = tf.reshape(coords, [-1, 2])
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
                    select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                    rays_o = tf.gather_nd(rays_o, select_inds)
                    rays_d = tf.gather_nd(rays_d, select_inds)
                    batch_rays = tf.stack([rays_o, rays_d], 0)
                    target_s = tf.gather_nd(target, select_inds)

            #####  Core optimization loop  #####

            with tf.GradientTape() as tape:

                # Make predictions for color, disparity, accumulated opacity.
                obj = d
                rgb, disp, acc, extras = render(H, W, focal, obj, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True, **render_kwargs_train)

                # Compute MSE loss between predicted and true RGB.
                img_loss = img2mse(rgb, target_s)
                trans.append(extras['raw'][..., -1])
                loss.append(img_loss)
                psnr.append(mse2psnr(img_loss))

                # Add MSE loss for coarse-grained model
                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_s)
                    loss[d] += img_loss0
                    psnr0.append(mse2psnr(img_loss0))

            gradients = tape.gradient(loss[d], grad_vars)
            optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:
            for d in range(args.diversity):
                obj = d
                rgbs, disps = render_path(render_poses[d], hwf[d], obj, args.chunk, render_kwargs_test)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_{}_'.format(expname, i, d))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

                if args.use_viewdirs:
                    render_kwargs_test['c2w_staticcam'] = render_poses[d][0][:3, :4]
                    rgbs_still, _ = render_path(render_poses[d], hwf[d], obj, args.chunk, render_kwargs_test)
                    render_kwargs_test['c2w_staticcam'] = None
                    imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            for d in range(args.diversity):
                obj = d
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}_{}'.format(i, d))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[d][i_test[d]].shape)
                render_path(poses[d][i_test[d]], hwf[d], obj, args.chunk, render_kwargs_test, gt_imgs=images[d][i_test[d]], savedir=testsavedir)
                print('Saved test set')
        
        if i % args.i_print == 0 or i < 10:
            print(expname, i, tensorlistmean(psnr).numpy(), tensorlistmean(loss).numpy())
            print('iter time {:.05f}'.format(dt))
            '''if i%args.i_print==0:
                tf.summary.scalar('loss', tensorlistmean(loss))
                tf.summary.scalar('psnr', tensorlistmean(psnr))
                tf.summary.histogram('tran', tensorlistmean(trans))
                if args.N_importance > 0:
                    tf.summary.scalar('psnr0', tensorlistmean(psnr0))'''

            if i % args.i_img == 0:

                # Log a rendered validation view to Tensorboard
                time0 = time.time()
                for d in range(args.diversity):
                    H, W, focal = hwf[d]
                    img_i = np.random.choice(i_val[d])
                    target = images[d][img_i]
                    pose = poses[d][img_i, :3, :4]
                    
                    obj = d

                    rgb, disp, acc, extras = render(H, W, focal, obj, chunk=args.chunk, c2w=pose, **render_kwargs_test)

                    psnr = mse2psnr(img2mse(rgb, target))
                    
                    # Save out the validation image for Tensorboard-free monitoring
                    testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                    if i==0:
                        os.makedirs(testimgdir, exist_ok=True)
                    imageio.imwrite(os.path.join(testimgdir, '{:06d}_{}.png'.format(i, d)), to8b(rgb))

                    '''
                    tf.summary.image('rgb_{}'.format(d), to8b(rgb)[tf.newaxis])
                    tf.summary.image('disp_{}'.format(d), disp[tf.newaxis, ..., tf.newaxis])
                    tf.summary.image('acc_{}'.format(d), acc[tf.newaxis, ..., tf.newaxis])
                    tf.summary.scalar('psnr_holdout_{}'.format(d), psnr)
                    tf.summary.image('rgb_holdout_{}'.format(d), target[tf.newaxis])

        tf.summary.experimental.set_step(i)'''


if __name__ == '__main__':
    train()