# helper functions for tensorflow

import re
import tensorflow as tf

def weight_variable(shape, name, seed):
    ''' Helper function to create TensorFlow variable

    Parameters:
    -----------
    shape, array_like
        shape of the returned variable
    name, str
        variables name
    seed, int
        random seed for the truncated_normal distribution

    Returns:
    --------
    variable tensor
    '''
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    ''' Helper function to create TensorFlow variable for biases

    Parameters:
    -----------
    shape, array_like
        shape of the returned variable
    name, str
        variables name

    Returns:
    --------
    variable tensor
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    ''' Helper function to calculate 2-dimensional convolution

    Parameters:
    -----------
    x, tensor
        data tensor
    W, tensor
        filter tensor

    Returns:
    --------
    tensor
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2d(x):
    ''' Helper function to perform max-pooling in 2d

    Parameters:
    -----------
    x, tensor
        data tensor

    Returns:
    --------
    tensor
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv3d(x, W):
    ''' Helper function to calculate 3-dimensional convolution

    Parameters:
    -----------
    x, tensor
        data tensor
    W, tensor
        filter tensor

    Returns:
    --------
    tensor
    '''
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_3d(x):
    ''' Helper function to perform max-pooling in 3d

    Parameters:
    -----------
    x, tensor
        data tensor

    Returns:
    --------
    tensor
    '''
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')

def activation_summary(x,tower_name):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
      tower_name: str
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % tower_name, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

