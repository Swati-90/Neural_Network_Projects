# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:07:45 2021

@author: lenovo
"""
# Loaidng libraries

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras import backend as K

# Loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Shape of X_train and y_train are", X_train.shape, y_train.shape)
print("Shape of X_test and y_test are", X_test.shape, y_test.shape)


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Convert class vector to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Shape of X_train and X_test is", X_train.shape, X_test.shape)

print("Train Samples", X_train.shape[0])
print("Test Samples", X_test.shape[0])

# Model Creation
batch_size = 128
num_classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), 
              metrics=['accuracy'])

# Model Training

hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                 verbose=1, validation_data=(X_test, y_test))

print("The model has trained successfully!!!")

model.save('mnist.h5')
print("Saving the model as mnist.h5")

# Model Evaluation

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy", score[1])

















