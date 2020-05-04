import tensorflow as tf

def dense(x, W, B, activation=None):
    if activation:
        return activation(tf.matmul(x, W) + B)
    return tf.matmul(x, W) + B


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(tf.random_normal_initializer([in_features, out_features]), name='dense_w')
        self.b = tf.Variable(tf.random_normal_initializer([out_features]), name='dense_b')
        self.name = name

    def __call__(self, t):
        y = tf.matmul(t, self.w) + self.b
        return tf.nn.relu(y)

class Conv2D():
    def __init__(self, kernel, in_channels, out_channels, strides, activation=None, name=None):
        super(Conv2D, self).__init__(name=name)
        self.w = tf.Variable(
            tf.random_normal_initializer([kernel[0], kernel[1], in_channels, out_channels]),
            name='conv2d_w')
        self.b = tf.Variable(
            tf.random_normal_initializer([64], name='conv2d_b'))
        self.activation = activation
        self.strides = strides
        self.name = name

    def __call__(self, t):
        y = tf.nn.conv2d(t, self.w, self.strides, 'VALID', )
