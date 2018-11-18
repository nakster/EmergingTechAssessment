# When the script is run on the computer you need to download keras with anaconda 
# copy and paste this line in the terminal 
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

# normalize inputs from 0-255 to 0-1
X_train/=255
X_test/=255

# convert class vectors to binary class matrices
#number of classes
classes = 10
# one-hot encoding
# we are expecting output as 8 means value of output variable 8
# so according to one-hot coding its [0,0,0,0,0,0,0,0,1,0]
y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)


# create model
model = Sequential()
# 32 = convolution filters to use, 5 =rows in each convolution kernel,
# and 5 = columns in each convolution kernel
# input_shape = (depth, width, height)
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Fit the model To fit the model, all we have to do is declare the batch size and number of epochs to train for, then pass in our training data.
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1000)

# Save the model to use test the pictures for later
model.save('models/mnistModel.h5')

# Final evaluation of the model
score  = model.evaluate(X_test, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# links https://anaconda.org/conda-forge/kerasC
# https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# https://keras.io/models/sequential/



