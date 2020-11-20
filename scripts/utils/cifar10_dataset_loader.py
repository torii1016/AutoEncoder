import argparse
import numpy as np
import pickle

import cv2
from utils.dataset_loader import Dataset

class Cifar10Dataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self._data_folder_path = config["path"]

        self._loading_dataset()


    def _get_input_data(self, filename, rows, cols, channels, classnum):

        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding="bytes")
    
        data = dict[b'data']
        labels = np.array(dict[b'labels'])

        if labels.shape[0] != data.shape[0]:
            raise Exception('Error: Different length')
        num_images = labels.shape[0]

        data = data.reshape(num_images, channels, rows, cols)
        data = data.transpose([0,2,3,1])
        data = np.multiply(data, 1.0/255.0)

        labels = self._dense_to_one_hot(labels, classnum)

        return data, labels


    def _dense_to_one_hot(self, labels_dense, num_classes):

        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot


    def _loading_dataset(self):

        images1, labels1 = self._get_input_data(self._data_folder_path+"data_batch_1", self._image_width, self._image_height, self._image_channel, self._output_class)
        images2, labels2 = self._get_input_data(self._data_folder_path+"data_batch_2", self._image_width, self._image_height, self._image_channel, self._output_class)
        images3, labels3 = self._get_input_data(self._data_folder_path+"data_batch_3", self._image_width, self._image_height, self._image_channel, self._output_class)
        images4, labels4 = self._get_input_data(self._data_folder_path+"data_batch_4", self._image_width, self._image_height, self._image_channel, self._output_class)
        images5, labels5 = self._get_input_data(self._data_folder_path+"data_batch_5", self._image_width, self._image_height, self._image_channel, self._output_class)
        test_images, test_labels = self._get_input_data(self._data_folder_path+"test_batch", self._image_width, self._image_height, self._image_channel, self._output_class)

        self._images = np.concatenate((images1, images2, images3, images4, images5), axis=0)
        self._labels = np.concatenate((labels1, labels2, labels3, labels4, labels5), axis=0)

        self._test_images =  self._images[:self._test_data_num]
        self._test_labels =  self._labels[:self._test_data_num]
        self._train_images = self._images[self._test_data_num:]
        self._train_labels = self._labels[self._test_data_num:]