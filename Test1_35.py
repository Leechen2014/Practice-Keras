# -*- coding: utf-8 -*-
# @File  : Test1_3.py
# @Author: lizhen
# @Date  : 2019/1/30
# @Desc  :

import  numpy as np
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.layers.core import Dropout
import pydot

import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import plot_model

# 设置超参数
NB_EPOCH=20

BATCH_SIZE=128
VERBOSE=1
NB_CLASSES =10
OPTIMIZER=SGD() # 优化器
N_HIDDEN=128
VALIDATION_SPLIT=0.2 # 训练数据集合中用于验证集的数据比例
DROPOUT=0.3 # 设置dropout 的失活率

# 数据准备
(X_train,Y_train),(X_test,Y_test) = mnist.load_data() #[NHWC]
data_SHAPE=28*28
train_num=60000
test_num=10000
# change to [N,WHC]
X_train =X_train.reshape(train_num,data_SHAPE).astype('float32')
X_test = X_test.reshape(test_num,data_SHAPE).astype('float32')
# 数据归一化
X_train /=255
X_test /=255
#
print(X_train.shape[0],'train samples') #get N
print(X_test.shape[0],'test sample')

# change label
Y_train = np_utils.to_categorical(Y_train,NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test,NB_CLASSES)

model = Sequential()
model.add(Dense(NB_CLASSES,input_shape=(data_SHAPE,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax')) ###

model.summary() # 打印出模型概况
# plot_model(model,to_file='test1_3.png',show_layer_names=True) # 绘制模型

model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])

his = model.fit(X_train,Y_train,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                verbose=VERBOSE,
                validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test,Y_test,verbose=VERBOSE)
print('Test score:',score[0])
print('Test accuracy',score[1])

print(his.history) # val_loss, val_acc,loss,acc
###
result=his.history
val_loss = result["val_loss"]
val_acc = result["val_acc"]
loss = result["loss"]
acc = result["acc"]

plt.title('Result Analysis')
plt.plot(val_loss, color='red', label='val_loss')
plt.plot(val_acc, color='blue', label='val_acc')
plt.plot(loss, color='green', label='loss')
plt.plot(acc, color='black', label='acc')

plt.legend()
plt.xlabel('opechs')
plt.ylabel('values')

# plt.show()
# fig =plt.gcf() # get current figure
# fig.savefig('data_result.png',dpi=100)
# or
plt.savefig("data_result.png")
plt.show()