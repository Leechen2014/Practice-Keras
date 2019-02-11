# -*- coding: utf-8 -*-
# @File  : show_data.py
# @Author: lizhen
# @Date  : 2019/1/30
# @Desc  :
import pandas as pd
import matplotlib.pyplot as plt

loss = pd.read_csv("/home/lizhen/workspace/Demesh/logger/loss1_log")
loss1 = loss["loss1"]

print(len(loss1))
total_loss = loss["toatal_loss"]

plt.title('Result Analysis')
plt.plot(loss1, color='red', label='loss')
plt.plot(total_loss, color='blue', label='total_loss')

plt.legend()
plt.xlabel('loss')
plt.xlabel('iterations')
plt.ylabel('loss values')

plt.show()