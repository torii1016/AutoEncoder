# -*- coding:utf-8 -*-

import tensorflow as tf
from .variable_util import get_const_variable, get_rand_variable
import numpy as np

def trans(name, inputs, k=3):

    in_channel = inputs.get_shape()[-1]
    weights = get_const_variable(name+"_weight", [in_channel, k*k], 0.0)
    biases = get_const_variable(name+"_biases", [k*k], 0.0)
    
    biases += tf.constant(np.eye(k).flatten(), dtype=tf.float32)
    
    transform = tf.matmul(inputs, weights)
    
    return tf.nn.bias_add(transform, biases)