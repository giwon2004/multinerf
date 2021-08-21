import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json

# Arguments

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

# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.math.log(x)/tf.math.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def tensorlistmean(tensorlist):
    s = 0
    for t in tensorlist:
        s += t
    return s / len(tensorlist)

# Positional encoding

class Embedder:

    def __init__(self, **kwargs):
    
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):

    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

# Ray helpers

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples