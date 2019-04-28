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




# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

pkfile = '/home/longlongaaago/WorkSpace/pycharm/pycharm_cnn_v1/cnn_test_ver.pk'

# load=True
# start = time.time()
# cnn_model_train_test_pk(pkfile,load,X_train, Y_train,X_test,Y_test,reg=0.001, learning_rate = 0.02, num_epochs = 100, minibatch_size = 16, print_cost = True)
# end = time.time()
# print end-start
parameters = initialize_parameters_pk(pkfile,True)
minibatch_size = 2
accuracy = test_model(X_test, Y_test, parameters,minibatch_size)