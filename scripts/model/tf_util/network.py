# -*- coding:utf-8 -*-

import os
import tensorflow as tf

from .conv import conv
from .deconv import deconv
from .linear import linear
from .batch_normalize import batch_norm
from .transform import trans

def fully_connection(inputs, 
                     scope,  name, 
                     output_channels, 
                     bn=False,
                     activation_fn=tf.nn.relu,
                     dropout=False,
                     drate=0.7,
                     is_training=False,
                     reuse=False):

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        outputs = linear(name=name,
                         inputs=inputs,
                         out_dim=output_channels)

        if bn:
            outputs = batch_norm(name, outputs, is_training)
            #outputs = tf.layers.batch_normalization(outputs, name=name, training=is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        if dropout:
            outputs = tf.nn.dropout(outputs, drate)
        
    return outputs


def conv2d(inputs, 
           scope,  name, 
           output_channels, 
           filter_size,
           stride=1,
           padding='SAME',
           bn=False,
           activation_fn=tf.nn.relu,
           is_training=False,
           reuse=False):

    with tf.compat.v1.variable_scope(scope, reuse=reuse):

        outputs = conv(inputs=inputs,
                       out_num=output_channels,
                       filter_width=filter_size[0], filter_height=filter_size[1],
                       stride=stride, 
                       name=name,
                       padding=padding)
        
        if bn:
            outputs = batch_norm(name, outputs, is_training)
            #outputs = tf.layers.batch_normalization(outputs, name=name, training=is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
    return outputs


def deconv2d(inputs, 
           scope,  name, 
           output_shape, 
           filter_size,
           stride=1,
           padding='SAME',
           bn=False,
           activation_fn=tf.nn.relu,
           is_training=False,
           reuse=False):

    with tf.compat.v1.variable_scope(scope, reuse=reuse):

        outputs = deconv(inputs=inputs,
                         out_shape=output_shape,
                         filter_width=filter_size[0], filter_height=filter_size[1],
                         stride=stride, 
                         name=name,
                         padding=padding)
        
        if bn:
            outputs = batch_norm(name, outputs, is_training)
            #outputs = tf.layers.batch_normalization(outputs, name=name, training=is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
    return outputs




def max_pool(inputs, 
             karnel_size=[1,1,1,1],
             strides=[1,1,1,1],
             padding='SAME'):

    return tf.nn.max_pool2d(inputs, ksize=karnel_size, strides=strides, padding=padding)


def transform(inputs,
              scope, name,
              k,
              reuse=False):

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        matrix = trans(name, inputs, k=k)
        
    return matrix

