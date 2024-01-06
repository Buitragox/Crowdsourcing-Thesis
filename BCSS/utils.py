import tensorflow as tf
import numpy as np
import pandas as pd
import skimage.io as io
from sklearn.utils import shuffle
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
    """
    Search the image folder and get the paths and onehot labels for all images.
    There should be a folder for each class. Ej: images/1/, images/2/, images/3/
    """
    
    ids = [] #paths for each image
    labels = [] #onehot labels 
    
    for root, dirs, files in os.walk(path):
        for i in range(len(files)):
            ids.append(os.path.join(root, files[i]))
            labels.append(int(root[-1]))

    ids = np.array(ids)
    labels = np.array(labels) - 1
    labels_onehot = tf.one_hot(labels, 3, on_value=1.0, off_value=0.0).numpy()

    return ids, labels_onehot


def load_labels_gold(pkl_path):
    """Read train_crowdsourced_labels.pkl and get the gold labels with onehot encoding"""
    df = pd.read_pickle(pkl_path)
    labels = df["label"] - 1 # -1 so labels start at 0
    labels_onehot = tf.one_hot(labels, 3, on_value=1.0, off_value=0.0).numpy()
    return labels_onehot


def load_labels_mv(pkl_path):
    """Read mv.pkl and get the annotations"""
    df = pd.read_pickle(pkl_path)
    labels = df["mv"] - 1 # -1 so labels start at 0
    labels_onehot = tf.one_hot(labels, 3, on_value=1.0, off_value=0.0).numpy() #majority voting labels
    return labels_onehot


def load_pickle_mv(pkl_path, images_path):
    """Read mv.pkl and prepare the image paths and labels"""
    df = pd.read_pickle(pkl_path)
    ids = [os.path.join(images_path, str(int(label)), patch) for patch, label in zip(df["patch"], df["label"])] #paths for each image
    ids = np.array(ids)
    labels_onehot = load_labels_mv(pkl_path)
    return ids, labels_onehot


def load_labels_ma(pkl_path, R):
    """
    Read train_crowdsourced_labels.pkl and prepare the image paths and labels for multiple annotators
    Annotator IDs go from 5 to 24, so R = 20 (amount of annotators)
    """
    df = pd.read_pickle(pkl_path)

    N = len(df)
    labels = [[-1 for _ in range(R)] for _ in range(N)]
    id_offset = 5
    for i in range(N):
        annotations = df["annotations"][i]
        for ann in annotations:
            ann_id = int(ann[0]) - id_offset
            ann_data = ann[1] - 1
            labels[i][ann_id] = ann_data

    labels = np.array(labels)

    return labels


def load_pickle_ma(pkl_path, images_path, R):
    """
    Read train_crowdsourced_labels.pkl and prepare the image paths and labels for multiple annotators
    Annotator IDs go from 5 to 24, so R = 20 (amount of annotators)
    """
    df = pd.read_pickle(pkl_path)
    ids = [os.path.join(images_path, str(int(label)), patch) for patch, label in zip(df["patch"], df["label"])] #paths for each image
    ids = np.array(ids)
    labels = load_labels_ma(pkl_path, R)

    return ids, labels


def load_npy_data(npy_path):
    """"Load Train and Test data from npy data format"""
    X_train = np.load(f'{npy_path}/TrainData.npy')
    X_test = np.load(f'{npy_path}/TestData.npy')
    Y_test = np.load(f'{npy_path}/TestLabels.npy')

    return X_train, X_test, Y_test


def load_gold_data(npy_path, pkl_path):
    """Load gold standard data from npy files and apply shuffle"""
    labels = load_labels_gold(pkl_path)
    X_train, X_test, Y_test = load_npy_data(npy_path)
    X_train, labels = shuffle(X_train, labels, random_state=42)
    return X_train, labels, X_test, Y_test


def load_mv_data(npy_path, pkl_path):
    """Load majority voting data from npy files and apply shuffle"""
    labels = load_labels_mv(pkl_path)
    X_train, X_test, Y_test = load_npy_data(npy_path)
    X_train, labels = shuffle(X_train, labels, random_state=42)
    return X_train, labels, X_test, Y_test


def load_ma_data(npy_path, pkl_path, R, min_two_ann=True):
    """Load multiple annotator data from npy files and apply shuffle"""
    labels = load_labels_ma(pkl_path, R)
    X_train, X_test, Y_test = load_npy_data(npy_path)

    # If min_two_ann=True, use data with at least 2 annotations
    if min_two_ann:
        i_ann = np.where(labels != -1, 1, 0)

        sum_i_ann = np.sum(i_ann, axis=1)
        i_ma = np.where(sum_i_ann >= 2, 1, 0)

        X_train = X_train[i_ma == 1,:]
        labels = labels[i_ma == 1,:]

    X_train, labels = shuffle(X_train, labels, random_state=42)
    return X_train, labels, X_test, Y_test
