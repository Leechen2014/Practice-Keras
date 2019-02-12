# -*- coding: utf-8 -*-
# @File  : Test3_31.py
# @Author: lizhen
# @Date  : 2019/2/11
# @Desc  : sample cifar10

from keras.datasets import cifar10
from keras.utils import np_utils

from keras.optimizers import SGD,Adam,RMSprop
from keras.layers.core import Activation,Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

from keras.models import Sequential

# from tensorflow.keras.models import Sequential
IMG_CHANNELS=3
IMG_ROWS=32
IMG_COLS=32

BATCH_SIZE=128
NB_EPOCH=20
NB_CLASSES=10
VERBOSE=1
VALIDATION_SPLIT=0.2
OPTIM=RMSprop()

(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

print('X_train shape',X_train.shape)
print(X_train.shape[0],' train samples')
print(X_test.shape[0], ' test samples')

# one hot encoding
Y_train = np_utils.to_categorical(Y_train,NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test,NB_CLASSES)

# as float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#
X_train /=255
X_test /=255

model = Sequential()
model.add(Conv2D(32,(3,3),padding='same'))
input_shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
#
model.compile(loss='categotiacal_crossentropy',optimizer=OPTIM,metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE)
score=model.evaluate(X_test,Y_test)

print('test score: ', score[0])
print('test acc: ', score[1])

#
# save model
model_struct=model.to_json()
open('cifar10_architecture.json','w').write(model_struct)
model.save_weights('cifar10_weights.h5',overwrite=True)