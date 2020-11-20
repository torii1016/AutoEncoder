# -*- coding:utf-8 -*-

import os
import tensorflow as tf

from .network import fully_connection, conv2d, deconv2d, max_pool, transform

class NetworkCreater(object):
    def __init__(self, config, name_scope):
        self._creater = {"conv2d":self._conv2d_creater,
                        "deconv2d":self._deconv2d_creater,
                        "fc":self._fc_creater,
                        "reshape":self._reshape_creater,
                        "transform":self._transform_creater,
                        "maxpool":self._maxpool_creater}
        self._active_function_list = {"ReLU":tf.nn.relu, "Sigmoid":tf.math.sigmoid, "None":None}
        self._name_scope = name_scope
        self._output_trans_dim = 0
        self._model_start_key = config["network"]["model_start_key"]


    def _conv2d_creater(self, inputs, data, is_training=False, reuse=True):
        return conv2d(inputs=inputs,
                   scope=self._name_scope,
                   name=data["name"], 
                   output_channels=data["output_channel"],
                   filter_size=data["fileter_size"],
                   stride=data["stride"],
                   padding=data["padding"],
                   bn=data["bn"],
                   activation_fn=self._active_function_list[data["activation_fn"]],
                   is_training=is_training,
                   reuse=reuse)


    def _deconv2d_creater(self, inputs, data, is_training=False, reuse=True):
        return deconv2d(inputs=inputs,
                   scope=self._name_scope,
                   name=data["name"], 
                   output_shape=data["output_channel"],
                   filter_size=data["fileter_size"],
                   stride=data["stride"],
                   padding=data["padding"],
                   bn=data["bn"],
                   activation_fn=self._active_function_list[data["activation_fn"]],
                   is_training=is_training,
                   reuse=reuse)


    def _fc_creater(self, inputs, data, is_training=False, reuse=True):
        return fully_connection(inputs=inputs,
                   scope=self._name_scope,
                   name=data["name"], 
                   output_channels=data["output_channel"],
                   bn=data["bn"],
                   activation_fn=self._active_function_list[data["activation_fn"]],
                   dropout=data["dropout"],
                   drate=data["drate"],
                   is_training=is_training,
                   reuse=reuse)

    def _reshape_creater(self, inputs, data, is_training=None, reuse=None):
        return tf.reshape(inputs, data["shape"])

    def _transform_creater(self, inputs, data, is_training=False, reuse=True):
        self._output_trans_dim = data["K"]
        return transform(inputs=inputs,
                   scope=self._name_scope,
                   name=data["name"], 
                   k=data["K"],
                   reuse=reuse)

    def _maxpool_creater(self, inputs, data, is_training=None, reuse=None):
        return max_pool(inputs=inputs,
                   karnel_size=data["karnel_size"],
                   strides=data["stride"],
                   padding=data["padding"])


    def create(self, inputs, config, is_training=True, reuse=False):
        h = inputs
        for layer in list(config.keys())[self._model_start_key:]:
            h = self._creater[config[layer]["type"]](inputs=h,
                                                    data=config[layer],
                                                    is_training=is_training,
                                                    reuse=reuse)
        return h
    
    def get_transform_output_dim(self):
        return self._output_trans_dim