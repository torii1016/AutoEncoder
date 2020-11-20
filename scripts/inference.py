import argparse
import sys
import os
import numpy as np
import toml

import cv2
import tensorflow as tf

from model.autoencoder import AutoEncoder
from utils.dataset_maker import DatasetMaker

class Inferencer(object):
    def __init__(self, config, data_loader=None):
        autoencoder_config = toml.load(open(config["network"]))

        self._use_gpu = config["use_gpu"]
        self._saved_model_path = config["save_model_path"]
        self._saved_model_name = config["save_model_name"]
    
        self._data_loader = data_loader
        self._label_name = self._data_loader.get_label_name_list()
        self._image_width, self._image_height, self._images_channel = self._data_loader.get_image_info()

        self._auto_encoder = AutoEncoder(param=config, config=autoencoder_config,
                        image_info=self._data_loader.get_image_info())
        self._auto_encoder.set_model()

        if self._use_gpu:
            config_system = tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    per_process_gpu_memory_fraction=0.8,
                    allow_growth=True
                )
            )
        else:
            config_system = tf.compat.v1.ConfigProto(
                device_count = {'GPU': 0}
            )

        self._sess = tf.compat.v1.Session(config=config_system)
        self._saver = tf.compat.v1.train.Saver()
        self._saver.restore(self._sess, self._saved_model_path + "/" + self._saved_model_name)


    def inference(self):
        input_images, input_labels = self._data_loader.get_test_data()
        output_images = self._auto_encoder.test(self._sess, input_images)

        for num, (input_image, output_image) in enumerate(zip(input_images, output_images)):
            image = input_image*255
            cv2.imwrite("{}_input.png".format(num), image)
            image = output_image.reshape(28,28,1)*255
            cv2.imwrite("{}_output.png".format(num), image)


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/testing_param.toml", type=str, help="default: config/testing_param.toml")
    args = parser.parse_args()

    config = toml.load(open(args.config))
    mode = config["test"]["dataset"]

    data_loader = DatasetMaker()(mode=mode, config=config["dataset"][mode])
    data_loader.print()

    inferencer = Inferencer(config["test"], data_loader)
    inferencer.inference()