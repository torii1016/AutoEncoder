import argparse
import sys
import os
import numpy as np
import toml
from tqdm import tqdm
import pickle
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf

from model.autoencoder import AutoEncoder
from utils.dataset_maker import DatasetMaker


class Loss(object):
    def __init__(self):
        self._loss = {"loss":[]}
    
    def append(self, loss):
        self._loss["loss"].append(loss)

    def save_log(self, name="logs/loss"):
        ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(range(len(self._loss["loss"])), self._loss["loss"], color="r", label="loss", linestyle="-")
        ax.set_xlabel("episode")
        ax.set_ylabel("loss")
        ax.set_ylim(0, np.max(self._loss["loss"])+np.max(self._loss["loss"])/10.0)
        ax.grid()
        plt.savefig(name+".png")
        pickle.dump(self._loss, open(name+".pickle", "wb"))



class Trainer(object):
    def __init__(self, config, data_loader=None):
        autoencoder_config = toml.load(open(config["network"]))

        self._batch_size = config["batch_size"]
        self._epoch = config["epoch"]
        self._val_step = config["val_step"]
        self._use_gpu = config["use_gpu"]
        self._save_model_path = config["save_model_path"]
        self._save_model_name = config["save_model_name"]
    
        self._data_loader = data_loader
        self._label_name = self._data_loader.get_label_name_list()

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
        init = tf.compat.v1.global_variables_initializer()
        self._sess.run(init)
        self._saver = tf.compat.v1.train.Saver()

        self._accuracy = 0.0
        self._loss = Loss()
        self._tensorboard_path = "./logs/" + datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


    def _save_model(self):
        os.makedirs(self._save_model_path, exist_ok=True)
        self._saver.save(self._sess, self._save_model_path+"/"+self._save_model_name)


    def _save_tensorboard(self, loss):
        with tf.name_scope('log'):
            tf.compat.v1.summary.scalar('loss', loss)
            merged = tf.compat.v1.summary.merge_all()
            writer = tf.compat.v1.summary.FileWriter(self._tensorboard_path, self._sess.graph)


    def train(self):
        accuracy = 0
        with tqdm(range(self._epoch)) as pbar:
            for i, ch in enumerate(pbar): #train

                input_images, input_labels = self._data_loader.get_train_data(self._batch_size)
                loss, _ = self._auto_encoder.train(self._sess, input_images)
                pbar.set_postfix(OrderedDict(loss=loss, accuracy=accuracy))

                #self._save_tensorboard(loss)
                self._loss.append(loss)

                if i%self._val_step==0: #test
                    self._save_model()
                    self._loss.save_log()


if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Process some integers' )
    parser.add_argument( '--config', default="config/training_param.toml", type=str, help="default: config/training_param.toml")
    args = parser.parse_args()

    config = toml.load(open(args.config))
    mode = config["train"]["dataset"]

    data_loader = DatasetMaker()(mode=mode, config=config["dataset"][mode])
    data_loader.print()

    trainer = Trainer(config["train"], data_loader)
    trainer.train()