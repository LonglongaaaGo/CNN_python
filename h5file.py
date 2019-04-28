# coding: utf-8
import os
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import h5py
import scipy

#专门用于生成  h5  文件的脚本 很重要
#将图片压缩成 h5py  文件，方便训练和读取

# #测试数据集的根目录
# traindata_root = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/data_ori/train_data'
# #相应的label 的文件位置
# label = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/label.txt'
# #相应的文件映射位置
# train_map = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/train_map.txt'
#
# #最后的h5文件
# h5_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/datah5/cardata.h5'



# #创建label
# crate_label(traindata_root,label)
#
# #限制的数量
# count = 2000
# #创建文件映射
# create_map(traindata_root,label,train_map,count)
# #创建h5file
# creat_h5_file(train_map,h5_path)

#载入h5文件
# load_hdf5_dataset(h5_path)





# label 期望生成的目标label文件
# traindata_root 数据集路径
# 生成标签文件label.txt
def crate_label(traindata_root, label):
    filelist = os.listdir(traindata_root)
    label_file = file(label, 'w')

    count = 1
    for f in filelist:
        if (f == '0'):
            temp = '0' + ' ' + f + '\n'
            label_file.write(temp)

    for f in filelist:
        if (f == '0'):
            continue
        else:
            temp = str(count) + ' ' + f + '\n'
            count += 1
            label_file.write(temp)
    label_file.close()


# test test.txt 的全路径
# tag 对应的映射 如 大众 1
# filepath 想要递归的文件路径
# 递归得生成 数据 和 tag 的集  写入 目标test.txt文件中
# num 设置的最大数量，不可超过
def write_recursive_map(filepath, tag, test, num):
    filelist = os.listdir(filepath)
    for file in filelist:
        file_new_path = os.path.join(filepath, file)
        # print file_new_path
        if os.path.isdir(file_new_path):
            write_recursive_map(file_new_path, tag, test, num)
        else:
            if num[tag] <= 0:
                break
            n_line = file_new_path + ' ' + str(tag) + '\n'
            test.write(n_line)
            num[tag] -= 1


# 直接根据  train 的label 和 测试的数据 生成对应的标签对应集
# fileroot = '/home/yanggang/longlongaaago/mobienet_v2/test_data'#数据集root
# label = '/home/yanggang/longlongaaago/mobienet_v2/label.txt' #标签
# 准备生成的标签对应集的文件
# text_file = '/home/yanggang/longlongaaago/mobienet_v2/test_label/test_test.txt'
# count 产生的数量，每个正类限制为 count 张 ，负类为 2.5*count 张
def create_map(fileroot, label, text_file, count):
    filelist = os.listdir(fileroot)
    label = open(label, 'r')
    test = file(text_file, 'w')

    lenth = 0

    add = 1

    num = {}
    while 1:
        line = label.readline()
        if not line:
            break
        lenth += 1
        temp = line.split()
        if temp[0] == 0:
            add = 0
            #                 print tag


    for i in range(lenth):
        for f_ in filelist:
            tag = 0
            label.seek(0)
            while 1:
                line = label.readline()
                if not line:
                    break
                temp = line.split()
                if f_ == temp[1]:
                    tag = temp[0]
                    #                 print tag
                    break
            label.seek(0)

            filepath = os.path.join(fileroot, f_)

            #             print '=---tag-----:'+str(tag)
            #             print '=----i-----:'+str(i)
            if os.path.isdir(filepath) and str(i+add) == str(tag):
                if (f_ == '0'):
                    num[tag] = count * 2.5
                else:
                    num[tag] = count

                write_recursive_map(filepath, tag, test, num)
                break

    test.close()
    label.close()


# train_map  训练的额map 文件
def read_map(train_map):
    train_map_file = file(train_map, 'r')
    car = []
    car_label = []

    while 1:
        line = train_map_file.readline()
        if not line:
            break
        temp = line.split()
        car.append(temp[0])
        car_label.append(temp[1])

    train_map_file.close()

    return car, car_label


# 打乱顺序
def shuffle_label(car, car_label):
    temp = np.array([car, car_label])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    car = list(temp[:, 0])
    car_label = list(temp[:, 1])
    car_label = [int(i) for i in car_label]

    return car, car_label


def get_test_num(car, car_label):
    return int(len(car) * 0.2)


def writeh5file(car, car_label, test_num, h5_path):
    Train_image = np.zeros((len(car) - test_num, 64, 64, 3))
    Train_label = np.zeros((len(car) - test_num, )).astype(np.int32)

    sub = 0
    if(min(car_label)==1):
        sub = 1
    Test_image = np.zeros((test_num, 64, 64, 3))
    Test_label = np.zeros((test_num, )).astype(np.int32)

    for i in range(len(car) - test_num):

        image = np.array(ndimage.imread(car[i], flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64))
        Train_image[i] = np.array(my_image)
        Train_label[i] = np.array(car_label[i]-sub)

    for i in range(len(car) - test_num, len(car)):
        image = np.array(ndimage.imread(car[i], flatten=False))
        my_image = scipy.misc.imresize(image, size=(64, 64))
        Test_image[i + test_num - len(car)] = np.array(my_image)
        Test_label[i + test_num - len(car)] = np.array(car_label[i]-sub)

        # Create a new file
    f = h5py.File(h5_path, 'w')
    f.create_dataset('X_train', data=Train_image)
    f.create_dataset('y_train', data=Train_label)
    f.create_dataset('X_test', data=Test_image)
    f.create_dataset('y_test', data=Test_label)
    f.close()


def load_hdf5_dataset(h5_path):
    # Load hdf5 dataset
    train_dataset = h5py.File(h5_path, 'r')
    train_set_x_orig = np.array(train_dataset['X_train'][:])  # your train set features
    train_set_y_orig = np.array(train_dataset['y_train'][:])  # your train set labels
    test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
    test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
    train_dataset.close()
    print(train_set_x_orig.shape)
    print(train_set_y_orig.shape)

    print(train_set_x_orig.max())
    print(train_set_x_orig.min())

    print(test_set_x_orig.shape)
    print(test_set_y_orig.shape)

    # 测试
    # plt.imshow(train_set_x_orig[400])
    print(train_set_y_orig[400])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig



#h5_path   目标存取的文件
#train_map  训练的额map 文件
def creat_h5_file(train_map,h5_path):
    car,car_label =read_map(train_map)
    car,car_label = shuffle_label(car,car_label)
    test_num = get_test_num(car,car_label)
    writeh5file(car,car_label,test_num,h5_path)







# #
# #测试数据集的根目录
# traindata_root = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/data_ori/test_data'
# #相应的label 的文件位置
# label = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/label.txt'
# #相应的文件映射位置
# test_map = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/test_map.txt'
#
# #最后的h5文件
# h5_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/datah5/car_test_data.h5'
#
#
#
# #创建label
# crate_label(traindata_root,label)
#
# #限制的数量
# count = 2000
# #创建文件映射
# create_map(traindata_root,label,test_map,count)
# #创建h5file
# creat_h5_file(test_map,h5_path)
#
# #载入h5文件
# load_hdf5_dataset(h5_path)
#



#
#测试数据集的根目录
traindata_root = '/home/yanggang/longlongaaago/pycharm_cnn_v1/CardatasetV2/data_ori/test_data'
# traindata_root = '/home/yanggang/longlongaaago/pycharm_cnn_v1/CardatasetV2/data_ori/train_data'
#相应的label 的文件位置
label = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/label_v3.txt'
#相应的文件映射位置
train_map = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/test_map_v3.txt'
# train_map = '/home/yanggang/longlongaaago/pycharm_cnn_v1/label/train_map_v3.txt'

#最后的h5文件
# h5_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/datah5/car_train_data_v3.h5'
h5_path = '/home/yanggang/longlongaaago/pycharm_cnn_v1/Cardataset/datah5/car_test_data_v3.h5'



#创建label
crate_label(traindata_root,label)

#限制的数量
count = 2000
#创建文件映射
create_map(traindata_root,label,train_map,count)
#创建h5file
creat_h5_file(train_map,h5_path)
#
# #载入h5文件
load_hdf5_dataset(h5_path)
