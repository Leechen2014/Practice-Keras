# -*- coding: utf-8 -*-
# @File  : Test3_12.py
# @Author: lizhen
# @Date  : 2019/1/30
# @Desc  :
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import SGD, Adam

from keras.datasets import mnist
from keras.utils import np_utils

seed = 7
np.random.seed(seed)
class LeNet:
    @staticmethod
    def forward(input_shape, classes):
        model = Sequential()
        # conv -->RELU --> POOL
        model.add(Conv2D(filters=20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # conv -->RELU --> POOL
        model.add(Conv2D(filters=50, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten layer
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        # softmax classify
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model


# net and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 2
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROW, IMG_CLOS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (1, IMG_ROW, IMG_CLOS)

# 混合并且划分训练和测试数据集合
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 设置了Keras将要使用的维度顺序，
# 也可以通过keras.backend.image_dim_ordering()来获取当前维度的顺序。
# 对于Keras而支持两种后端引擎对张量进行计算。
K.set_image_dim_ordering("th")

# 把训练数据看成float类型，并且归一化
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 修改输入的形状
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], ' test samples')

# Y
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

model = LeNet.forward(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,metrics=['accuracy'] )#设置是否返回acc
#           accuracy
result = model.fit(X_train, Y_train,
                   batch_size=BATCH_SIZE, #
                   epochs=NB_EPOCH,
                   verbose=VERBOSE, # 控制台的输出信息 分别有0，1，2
                   validation_split=VALIDATION_SPLIT, # 验证集的划分
                   )

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

print("Test score: ", score)
print("Test accuracy", score)

print(' 所有的历史数据： ')
print(result.history.keys()) #dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

# 汇总历史曲线
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss ')
plt.xlabel('epoch')
plt.legend(["train", "test"], loc='upper left')
plt.show()
