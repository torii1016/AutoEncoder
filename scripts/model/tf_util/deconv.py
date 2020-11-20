#! -*- coding:utf-8 -*-

import tensorflow as tf

from .variable_util import get_const_variable, get_rand_variable

def deconv(name, inputs, out_shape, filter_width, filter_height, stride, padding):

    # ** NOTICE: weight shape is [height, width, out_chanel, in_chanel] **
    out_channel = out_shape[-1]
    in_channel = inputs.get_shape()[-1]

    out_shape[0] = tf.shape(inputs)[0] # convert from -1 to batch_size
    weights_shape =  [filter_height, filter_width, out_channel, in_channel]
    weights = get_rand_variable(name,weights_shape, 0.02)
    
    biases = get_const_variable(name, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights,
                                      output_shape=out_shape,
                                      strides=[1, stride, stride, 1],
                                      padding=padding)
    return tf.nn.bias_add(deconved, biases)
