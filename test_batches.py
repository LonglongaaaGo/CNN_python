# coding: utf-8
from cnn_fun import *
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import scipy
from PIL import Image
from scipy import ndimage
from cnn_utils import *
from dnn_utils_v2 import *
from h5file import *

import time



#最后的分析文件  重要

#进行批量测试
def test_model_baches(X_test, Y_test, parameters, minibatch_size=1):
    '''
    X_test ,(number,n_h,n_w,n_c)
    Y_test, (classes,number)

    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X_test   测试样本(m, n_H0, n_W0, n_C0)
    Y_test   测试样本的标签(classes,number)
    parameters 卷积核参数
    minibatch_size  batch
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_accur -- True to print the cost every 100 epochs

    Returns:
    accuracy --
    """
    '''


    y_out = []
    y_hat_out = []

    np.random.seed(1)
    (m, n_H0, n_W0, n_C0) = X_test.shape

    print '-----m-----'
    print m
    seed = 1
    acc_num = 0

    # number of minibatches of size minibatch_size in the train set
    # num_minibatches = int(m / minibatch_size)
    minibatches = random_mini_batches(X_test, Y_test, minibatch_size, seed)

    for minibatch in minibatches:
        # Select a minibatch
        (minibatch_X, minibatch_Y) = minibatch
        acc_temp,y_,y_hat_ = test_model_onces_batches(minibatch_X, minibatch_Y, parameters)

        acc_num += acc_temp
        temp_ = minibatch_X.shape[0]

        print '---temp_acc-----'
        print '--' + str(float(acc_temp) / temp_)

        y_out.extend(y_)
        y_hat_out.extend(y_hat_)



    accuracy = float(acc_num) / m

    print '---accuracy--'
    print accuracy

    acc_parameter = {
        'y_out':y_out,
        'y_hat_out':y_hat_out
    }

    return accuracy,acc_parameter



def test_model_onces_batches(X_test, Y_test, parameters):
    '''
    X_test ,(number,n_h,n_w,n_c)
    Y_test, (classes,number)
    parameters  卷积核参数
    1次批次计算
    return :
    the true number of the X_test
    '''
    reg = 0.01
    loss, grad3, caches = forward_propagation_vectorized(X_test, parameters, Y_test, reg)

    cache_3 = caches['cache_3']

    AL = cache_3['y_hat']

    print '-----AL------'
    print AL
    # y_hat = Y_hat2y_hat(AL)
    y_hat = Y_hat2y_hat_ver(AL)

    y = Y_to_y(Y_test)
    print'------y_hat-----'
    print y_hat
    print'-------y------'
    print y



    reult = (y == y_hat)

    true_num = np.sum(reult)

    #     accuracy = float(true)/len(reult)

    #     print reult

    #     print '---accuracy--'
    #     print accuracy

    return true_num,y,y_hat




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
h5_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/datah5/car_test_data_v3.h5'
#将测试集文件夹的文件生成最终的h5 文件

#最后训练的参数
pkfile = '/home/yanggang/longlongaaago/pycharm_cnn_v1/out/outV3/cnnv3car_205.pk'
load =True
#参数读取
parameters = initialize_parameters_pk(pkfile, load)
# Loading the data (signs)
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_hdf5_dataset(h5_path)
#数据读取
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


# X_ = X_train[0:10,:,:,:]
# X_t = X_test[0:10,:,:,:]
# Y_ = Y_train[0:10]
# Y_t = Y_test[0:10]
# acc_test,acc_test_parameter = test_model_baches(X_t, Y_t, parameters, minibatch_size=4)
# acc_train,acc_train_parameter = test_model_baches(X_, Y_, parameters, minibatch_size=4)


#拼接操作，最初的时候生成 h5 文件时，自动得分成了  测试集和训练集 的字典，所以要合并一下

acc_train,acc_train_parameter = test_model_baches(X_train, Y_train, parameters, minibatch_size=16)
acc_test,acc_test_parameter = test_model_baches(X_test, Y_test, parameters, minibatch_size=16)
acc_train_y = acc_train_parameter['y_out']
acc_train_yhat = acc_train_parameter['y_hat_out']



acc_test_y = acc_test_parameter['y_out']
acc_test_yhat = acc_test_parameter['y_hat_out']

acc_train_y.extend(acc_test_y)
acc_train_yhat.extend(acc_test_yhat)


acc_train_parameter['y_out'] = acc_train_y
acc_train_parameter['y_hat_out'] = acc_train_yhat



print acc_train_y
print acc_train_yhat




#根据y 和yhat数组，进行分析
#acc_train_y  y数组
#acc_train_yhat   预判yhat 数组
#return 字典，给每个类的准确率进行统计
def calss_acc(acc_train_y,acc_train_yhat):
    acc_classer ={}

    print len(acc_train_y)

    for i in range(len(acc_train_y)):
        key = 'sum' + str(acc_train_y[i])
        if not key in acc_classer.keys():
            acc_classer['sum' + str(acc_train_y[i])] = 0

        key_true = 'true' + str(acc_train_y[i])
        if not key_true in acc_classer.keys():
            acc_classer['true' + str(acc_train_y[i])] = 0

        key_fales = 'false' + str(acc_train_yhat[i])
        if not key_fales in acc_classer.keys():
            acc_classer['false' + str(acc_train_yhat[i])] = 0


        acc_classer['sum' + str(acc_train_y[i])] = acc_classer['sum' + str(acc_train_y[i])] + 1
        if acc_train_y[i] == acc_train_yhat[i]:

            acc_classer['true' + str(acc_train_y[i])] = acc_classer['true' + str(acc_train_y[i])] + 1
        else:
            acc_classer['false' + str(acc_train_yhat[i])] = acc_classer['false' + str(acc_train_yhat[i])] + 1

        acc_key = 'acc_true' + str(acc_train_y[i])
        acc_classer[acc_key] = float(acc_classer['true' + str(acc_train_y[i])])/acc_classer['sum' + str(acc_train_y[i])]

        false_key = 'acc_false' + str(acc_train_y[i])
        acc_classer[false_key] = float(acc_classer['false' + str(acc_train_y[i])]) / acc_classer['sum' + str(acc_train_y[i])]

    return acc_classer


#label_path  label 的全路径
#return 按照顺序的名称列表
def read_label(label_path):

    name_list = []

    # filelist = os.listdir(traindata_root)
    label_file = file(label_path, 'r')

    while 1:
        line = label_file.readline()
        if not line:
            break
        temp = line.split()
        name_list.append(temp[1])

    return name_list

#label_path  label 的全路径
#acc_train_calss   关于测试结果的字典
#result_file   最终要保存的结果文件   .txt
def saveResult(label_path,acc_train_calss,result_file):
    result = file(result_file, 'w')

    name  = read_label(label_path)



    strs = '类别' + '\t' +'总数'+ '\t' + '正确 '+'\t' +'正确率 '+'\t' +'错误 '+'\t' +'错误率 '+'\n'
    result.write(strs)

    for i in range(len(name)):
        strs = name[i]+ '\t'+ str(acc_train_calss['sum'+str(i)])+'\t'+ str(acc_train_calss['true'+str(i)])+'\t'+ str(acc_train_calss['acc_true'+str(i)])+'\t'+ str(acc_train_calss['false'+str(i)])+'\t'+ str(acc_train_calss['acc_false'+str(i)])+'\n'
        result.write(strs)








#根据y 和yhat数组，进行分析
#return 字典，给每个类的准确率进行统计
acc_train_calss = calss_acc(acc_train_y,acc_train_yhat)

print acc_train_calss
#最后保存的结果存起来
acc_class_pk = '/home/yanggang/longlongaaago/pycharm_cnn_v1/result/car1_v3_class_acc.pk'

save_model(acc_train_calss,acc_class_pk)

#label_path   标签文件路径
label_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/label_v3.txt'
#最终的result  文件
result_file = '/home/yanggang/longlongaaago/pycharm_cnn_v1/result/car1_v3_class_result.txt'


saveResult(label_path,acc_train_calss,result_file)








