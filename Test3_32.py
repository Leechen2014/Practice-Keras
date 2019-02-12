# -*- coding: utf-8 -*-
# @File  : Test3_31.py
# @Author: lizhen
# @Date  : 2019/2/11
# @Desc  : sample cifar10

from keras.datasets import cifar10
from keras.utils import np_utils

from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt
from keras import metrics

# from tensorflow.keras.models import Sequential
NUM_TO_AUGENT=5

IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


print('X_train shape', X_train.shape)
print(X_train.shape[0], ' train samples')
print(X_test.shape[0], ' test samples')

# one hot encoding
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

# as float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#
X_train /= 255
X_test /= 255

# network
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same'))
input_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
model.add(Activation('relu'))
##### start ######
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(63, 3, 3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
##### end ######
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

#                   categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIM,
              metrics=[metrics.mae,metrics.categorical_accuracy])

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

result=model.fit(X_train, Y_train,
                 batch_size=BATCH_SIZE,
                 epochs=NB_EPOCH,
                 verbose=VERBOSE,
                 validation_split=VALIDATION_SPLIT)

# model.fit_generator(datagen.flow(X_train, Y_train,
#                        batch_size=BATCH_SIZE),
#                        samples_per_epoch=X_train.shape[0],
#                        nb_epoch=NB_EPOCH,
#                        verbose=VERBOSE)

#server.launch(model)
#
print('Testing ...')
score = model.evaluate(X_test, Y_test)
print('test score: ', score[0])
print('test acc: ', score[1])

mode_json=model.to_json()
open('cifar10_architecture.json','w').write(mode_json)
model.save_weights('cifar10_weights.h5',overwrite=True)

# list all data in history
print(result.history.keys())
# summarize history for accuracy
plt.plot(result.history['categorical_accuracy'])
plt.plot(result.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accruacy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

# summarize history for loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()