#本项目的入口，非常重要！！！
# coding: utf-8

import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import scipy
from PIL import Image
from scipy import ndimage
from cnn_utils import *
from dnn_utils_v2 import *
from cnn_fun import *
from h5file import *

import time
# % matplotlib
# inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# 
# % load_ext
# autoreload
# % autoreload
# 2

np.random.seed(1)



# #最后的h5文件
# h5_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/datah5/cardata.h5'

#v3  数据集变化 读入相应的文件
h5_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/datah5/car_train_data_v3.h5'

# Loading the data (signs)
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_hdf5_dataset(h5_path)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_hdf5_dataset(h5_path)
# Example of a picture
# index = 6
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = Y_train_orig
Y_test = Y_test_orig
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
# conv_layers = {}

# pkfile = '/home/yanggang/longlongaaago/pycharm_cnn_v1/out/cnn_v2_20180426.pk'
#v1
# pkfile = '/home/yanggang/longlongaaago/pycharm_cnn_v1/out/cnnv1car_68.pk'
# others_file = '/home/yanggang/longlongaaago/pycharm_cnn_v1/other_para/cnnv1carOtherparameter_0.pk'
#

#v2
# pkfile = '/home/yanggang/longlongaaago/pycharm_cnn_v1/out/outV2/cnnv2car_101.pk'
# others_file = '/home/yanggang/longlongaaago/pycharm_cnn_v1/other_para/cnnv2carOtherparameter_0.pk'
#

#v3
#上次训练保留下来的参数，做持久化训练
pkfile = '/home/yanggang/longlongaaago/pycharm_cnn_v1/out/outV4/cnnv4car1_38.pk'
#同事对应的  损失值，测试准确率  等文件保存
others_file = '/home/yanggang/longlongaaago/pycharm_cnn_v1/other_para/cnnv4carOtherparameter_0.pk'
#



# X_ = X_train[0:10,:,:,:]
# X_t = X_test[0:10,:,:,:]
# Y_ = Y_train[0:10]
# Y_t = Y_test[0:10]
#
load=True
start = time.time()
# cnn_model_train_test_pk(pkfile,load,X_train, Y_train,X_test,Y_test,reg=0.0, learning_rate = 0.005, num_epochs = 100, minibatch_size = 2, print_cost = True)

# save_speed = 5

train_main(others_file,pkfile, load,X_train, Y_train,X_test,Y_test,main_epoch=100,reg=0.000001, learning_rate = 0.002, num_epochs = 1, minibatch_size = 16, print_cost = True)

end = time.time()
print end-start







