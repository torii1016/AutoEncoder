import argparse
import numpy as np
import pickle

import cv2

class Dataset(object):
    def __init__(self, config):
        self._dataset_name = config["name"]
        self._test_data_num = config["test_data_num"]
        self._label_name = config["label_name"]

        self._image_width = config["image_width"]
        self._image_height = config["image_height"]
        self._image_channel = config["image_channels"]

        self._output_class = len(self._label_name)

        self._train_images = []
        self._train_labels = []
        self._test_images = []
        self._test_labels = []


    def _load_dataset(self):
        pass


    def get_train_data(self, batch_size=50):
        choice_data = np.random.choice(range(len(self._train_images)), batch_size, replace=False)
        return self._train_images[choice_data], self._train_labels[choice_data]


    def get_test_data(self):
        return self._test_images, self._test_labels


    def get_label_name_list(self):
        return self._label_name
    

    def get_image_info(self):
        return self._image_width, self._image_height, self._image_channel


    def print(self):
        print("--------------------")
        print("[Dataset]          {}".format(self._dataset_name))
        print("[Number of images] {}".format(len(self._images)))
        print("[Number of train]  {}".format(len(self._train_images)))
        print("[Number of test]   {}".format(len(self._test_images)))
        print("[Label list]       {}".format(self._label_name))
        print("[Images width]     {}".format(self._image_width))
        print("[Images height]    {}".format(self._image_height))
        print("[Images channels]  {}".format(self._image_channel))
        print("--------------------")