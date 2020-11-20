import os 
import sys

from utils.cifar10_dataset_loader import Cifar10Dataset
from utils.voc2007_dataset_loader import VOC2007Dataset
from utils.mnist_dataset_loader import MNISTDataset

class DatasetMaker(object):
    def __init__(self):
        self._dataset = {"Cifar10":Cifar10Dataset, "VOC2007":VOC2007Dataset, "MNIST":MNISTDataset}
    
    def __call__(self, mode, config):
        return self._dataset[mode](config)