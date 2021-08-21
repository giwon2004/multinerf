import tensorflow as tf

def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, input_ch_obj=1, input_obj_dim=5, output_ch_obj=5, output_ch=4, skips=[4], use_viewdirs=False):

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, input_obj_dim, 'to', output_ch_obj,
          type(input_ch), type(input_ch_views), type(input_ch_obj), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)
    input_ch_obj = int(input_ch_obj)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views + input_ch_obj))
    inputs_pts, inputs_obj, inputs_views = tf.split(inputs, [input_ch, input_ch_obj, input_ch_views], -1)
    outputs_obj = tf.keras.layers.Embedding(input_obj_dim, output_ch_obj)(inputs_obj)
    outputs_obj = tf.reduce_sum(outputs_obj, axis=1)
    inputs_pts = tf.concat([inputs_pts, outputs_obj], -1)
    inputs_pts.set_shape([None, input_ch + output_ch_obj])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat([bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model