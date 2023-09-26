import tensorflow as tf
import numpy as np
import pandas as pd
import skimage.io as io
import os

class DataSeq(tf.keras.utils.Sequence):

    def __init__(self, ids, labels, batch_size=8, image_size=224):
        self.ids = ids
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size

    def __load__(self, id_name):
        image_path = id_name
        image = io.imread(image_path) / 255.0

        return image

    def __getitem__(self, index):
        n = self.batch_size

        if (index + 1) * n > len(self.ids):
            n = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * n : (index + 1) * n]
        labels_batch = self.labels[index * n : (index + 1) * n]
        images = []
        
        for id in files_batch:
            image = self.__load__(id)
            images.append(image)
            
        images = np.array(images)
        labels_batch = np.array(labels_batch)

        return images, labels_batch

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))


def load_ids(path):
    ids = []
    labels = []
    
    for root, dirs, files in os.walk(path):
        for i in range(len(files)):
            ids.append(os.path.join(root, files[i]))
            labels.append(int(root[-1]))

    ids = np.array(ids)
    labels = np.array(labels) - 1
    labels_onehot = tf.one_hot(labels, 3, on_value=1.0, off_value=0.0).numpy()

    return ids, labels_onehot 