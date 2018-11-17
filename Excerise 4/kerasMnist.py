# conda install -c conda-forge keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

# load data
# X_train = training images 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# These are dimensions of the array [60000,23,28]
# print(X_train.shape[0])#number of images 
# print(X_train.shape[1])#width
# print(X_train.shape[2])#height

# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')



#links https://anaconda.org/conda-forge/kerasC
#https://elitedatascience.com/keras-tutorial-deep-learning-in-python