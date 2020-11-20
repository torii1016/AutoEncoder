import argparse
import numpy as np
import pickle

import cv2
from utils.dataset_loader import Dataset

class VOC2007Dataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self._dataset_image_path = config["image"]
        self._dataset_label_path = config["label"]

        self._default_image_width = 300
        self._default_image_height = 300

        self._images_with_box = []
        self._labels_with_box = []
        
        self._load_dataset()


    def _devide_by_box(self, img, labels):
        devided_image = []
        devided_label = []
        for label in labels:
            x_min = int(label[0]*self._default_image_width)
            y_min = int(label[1]*self._default_image_height)
            x_max = int(label[2]*self._default_image_width)
            y_max = int(label[3]*self._default_image_height)

            object_img = img[y_min:y_max, x_min:x_max]
            if 50<=object_img.shape[0] and 50<=object_img.shape[1]:
                devided_image.append(cv2.resize(object_img, (self._image_height, self._image_width)))
                devided_label.append(label[4:])
        return devided_image, devided_label


    def _loading_images(self):

        images = []
        labels = []
        image_index = []
        for img_name in self.keys:
            img = self.loading_image(self._dataset_image_path+"/"+img_name)

            object_img, object_label = self._devide_by_box(img, self.label[img_name])
            images.extend(object_img)
            labels.extend(object_label)
            image_index.extend([len(self._images_with_box)]*len(object_img))

            self._images_with_box.append(img)
            self._labels_with_box.append(self.label[img_name])
        
        return np.array(images), np.array(labels), np.array(image_index)


    def loading_image(self, path):
        img = cv2.imread(path)
        h, w, c = img.shape
        img = cv2.resize(img, (self._default_image_width, self._default_image_height))
        img = img[:, :, ::-1].astype("float32")
        img /= 255
        return img



    def _loading_pickle_label(self):
        with open(self._dataset_label_path, "rb") as file:
            self.label = pickle.load(file)
            self.keys = sorted(self.label)


    def _load_dataset(self):
        self._loading_pickle_label()
        self._images, self._labels, self._image_index = self._loading_images()

        self._test_images = self._images[-self._test_data_num:]
        self._test_labels = self._labels[-self._test_data_num:]
        self._train_images = self._images[:-self._test_data_num]
        self._train_labels = self._labels[:-self._test_data_num]