# -*- coding:utf-8 -*-

import tensorflow as tf
from .variable_util import get_const_variable, get_rand_variable
from .sn import spectral_norm

def conv(name, inputs, out_num, filter_width, filter_height, stride, padding='SAME'):

    # ** NOTICE: weight shape is [height, width, in_chanel, out_chanel] **
    
    in_channel = inputs.get_shape()[-1]
    weights_shape = [filter_height, filter_width, in_channel, out_num]
    weights = get_rand_variable(name, weights_shape, 0.02)

    biases = get_const_variable(name, [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = padding)
    
    return tf.nn.bias_add(conved, biases)

def sn_conv(name, inputs, out_num, filter_width, filter_height, stride, update_collection = 'SN_UPDATE_OP'):

    # ** NOTICE: weight shape is [height, width, in_chanel, out_chanel] **
    
    in_channel = inputs.get_shape()[-1]
    weights_shape = [filter_height, filter_width, in_channel, out_num]
    weights = get_rand_variable(name, weights_shape, 0.02)

    biases = get_const_variable(name, [out_num], 0.0)

    W_shape = weights.shape.as_list()
    u = tf.get_variable("u_{}".format(name), [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    
    conved = tf.nn.conv2d(inputs, spectral_norm(weights, u=u, update_collection = update_collection),
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)
