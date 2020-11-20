import argparse
import numpy as np
import pickle

import cv2
from sklearn.datasets import fetch_openml
from utils.dataset_loader import Dataset

class MNISTDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self._data_folder_path = config["path"]

        self._load_dataset()


    def _load_dataset(self):

        mnist = fetch_openml(self._data_folder_path, version=1,)
        self._images = mnist.data.astype(np.float32)
        self._images /= 255 
        self._labels = mnist.target.astype(np.int32)
 
        self._train_images, self._test_images = np.split(self._images,   [len(self._images)-self._test_data_num])
        self._train_labels, self._test_labels = np.split(self._labels, [len(self._labels)-self._test_data_num])

        self._train_images = self._train_images.reshape((len(self._train_images), self._image_height, self._image_width, self._image_channel)) # (N, height, width, channel)
        self._test_images = self._test_images.reshape((len(self._test_images), self._image_height, self._image_width, self._image_channel))

        self._train_labels = np.eye(np.max(self._train_labels)+1)[self._train_labels]
        self._test_labels = np.eye(np.max(self._test_labels)+1)[self._test_labels]