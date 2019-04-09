import numpy as np
import keras
import cv2 as cv2
from keras.applications.resnet50 import preprocess_input
from os.path import join

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, img_dir, batch_size=32, target_shape=(500,500), n_channels=3,
                 n_classes=25, shuffle=True):
        'Initialization'
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.target_shape, self.n_channels))
        #print(X.shape)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#             print(join(self.img_dir,ID))
#             print(self.target_shape)
            img = cv2.imread(join(self.img_dir,ID))
            img = cv2.resize(img, self.target_shape, cv2.INTER_CUBIC)
            X[i,] = img


        return preprocess_input(X), ID