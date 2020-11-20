# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import pickle

from .tf_util import Layers, NetworkCreater, cross_entropy

class _autoencoder_network(Layers):
    def __init__(self, name_scopes, config):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)

        self._config = config["AutoEncoder"]
        self._network_creater = NetworkCreater(self._config, name_scopes[0]) 

    def set_model(self, inputs, is_training=True, reuse=False):
        return self._network_creater.create(inputs, self._config, is_training, reuse)


class AutoEncoder(object):
    
    def __init__(self, param, config, image_info):
        self._lr = param["lr"]
        self._image_width, self._image_height, self._image_channels = image_info

        self._network = _autoencoder_network([config["AutoEncoder"]["network"]["name"]], config)


    def set_model(self):
        self._set_network()
        self._set_loss()
        self._set_optimizer()


    def _set_network(self):
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self._image_height, self._image_width, self._image_channels])

        self._logits = self._network.set_model(self.input, is_training=True, reuse=False) # train
        self._logits_wo = self._network.set_model(self.input, is_training=False, reuse=True) # inference


    def _set_loss(self):
        loss = cross_entropy(tf.reshape(self.input, [-1, self._image_width*self._image_height*self._image_channels]), self._logits)
        self._loss_op = tf.reduce_mean(loss)


    def _set_optimizer(self):
        #self._train_op = tf.compat.v1.train.RMSPropOptimizer(self._lr).minimize(self._loss_op, var_list=self._network.get_variables())
        self._train_op = tf.compat.v1.train.AdamOptimizer(self._lr).minimize(self._loss_op)


    def train(self, sess, input_images):
        feed_dict = {self.input: input_images}
        loss, _, logits = sess.run([self._loss_op, self._train_op, self._logits], feed_dict=feed_dict)
        return loss, logits

    def test(self, sess, input_images):
        feed_dict = {self.input: input_images}
        logits = sess.run([self._logits_wo], feed_dict=feed_dict)
        return np.squeeze(logits)  # remove extra dimension