# coding: utf-8
from cnn_fun import *
#文件用途，最后分析使用！！
#不要随便解开注释
#
#other_file = 每一次epoch 所生成的 accuracy的一个list
#里面包含：
#other_parameters['costs']  
#other_parameters['acc_test']
#other_parameters['acc_train']
#分别为每次epoch 的损失值、验证机准确率、训练集准确率

# other_file= '/home/yanggang/longlongaaago/pycharm_cnn_v1/other_para/cnnv1carOtherparameter_0.pk'
#
# other_parameters = load_model(other_file)
#
# costs_v1 = other_parameters['costs']
# acc_test_v1 = other_parameters['acc_test']
# acc_train_v1 = other_parameters['acc_train']


#数据错乱的处理
# cost_v2 = costs_v1[0:68]
# acc_test_v2 = acc_test_v1[0:68]
# acc_train_v2 = acc_train_v1[0:68]
#
# cost_v2.extend(costs_v1[138:154])
# acc_test_v2.append(acc_test_v1[138:154])
# acc_train_v2.append(acc_train_v1[138:154])


#
# others_file_v2 = '/home/yanggang/longlongaaago/pycharm_cnn_v1/other_para/cnnv2carOtherparameter_0.pk'
#
# other_parameters = load_model(others_file_v2)
# costs_v2 = other_parameters['costs']
# acc_test_v2 = other_parameters['acc_test']
# acc_train_v2 = other_parameters['acc_train']

#处理数据混乱
# acc_test_v2_ = acc_test_v2[0:68]
# # print acc_test_v2[68]
# acc_test_v2_.extend(acc_test_v2[68])
# acc_test_v2_.extend(acc_test_v2[69:])
#
#
# print acc_test_v2_
# print acc_test_v2
#
#
#
# acc_train_v2_ = acc_train_v2[0:68]
# # print acc_test_v2[68]
# acc_train_v2_.extend(acc_train_v2[68])
# acc_train_v2_.extend(acc_train_v2[69:])
#
#
# print acc_train_v2_
# print acc_train_v2




def save_otherPar(cost_array,acc_test_array,acc_train_array,others_file):
    other_parameters  = {
        'costs':cost_array,
        'acc_test' :acc_test_array,
        'acc_train':acc_train_array
    }
    save_model(other_parameters, others_file)

# save_otherPar(costs_v2,acc_test_v2_,acc_train_v2_,others_file_v2)

#删除多余的错误的数据
def deleteSurplus(cost_array,acc_test_array,acc_train_array,others_file):
    cost_ok = cost_array[0:21]
    acctest = acc_test_array[0:21]
    acc_train = acc_train_array[0:21]
    save_otherPar(cost_ok,acctest,acc_train,others_file)

# deleteSurplus(costs_v1,acc_test_v1,acc_train_v1,other_file)



#绘制损失曲线
def draw_cost(costs_main,version):
    # plot the cost
    plt.plot(np.squeeze(costs_main))
    plt.ylabel('cost')
    plt.xlabel('iterations (per epoch)')
    # plt.title("Learning rate =" + str(0.005))
    plt.title(version+':'+"Cost per epoch" )

    plt.show()

# draw_cost(costs_v2,'v2')
# print costs_v2
# draw_cost(costs_v2)
# print costs_v2


#绘制准确率曲线
def draw_acc_train(acc_train,version):
    # plot the cost

    plt.plot(np.squeeze(acc_train))
    plt.ylabel('train_accuracy')
    plt.xlabel('iterations (per epoch)')
    plt.title(version+':'+"train_accuracy per epoch")

    plt.show()

# draw_acc_train(acc_train_v2,'v2')
# print acc_train_v2

#绘制测试集准确率曲线
def draw_acc_test(acc_test,version):
    # plot the cost
    plt.plot(np.squeeze(acc_test))
    plt.ylabel('test_accuracy')
    plt.xlabel('iterations (per epoch)')
    plt.title(version+':'+"test_accuracy per epoch")
    plt.show()

#
# draw_acc_test(acc_test_v2,'v2')
# print acc_test_v2

#return the best model
def get_best(cost_array,acc_test_array,acc_train_array):


    print np.min(cost_array)
    print np.where(cost_array == np.min(cost_array))[0]
    print np.max(acc_test_array)
    print np.where(acc_test_array == np.max(acc_test_array))[0]
    print np.max(acc_train_array)
    print np.where(acc_train_array == np.max(acc_train_array))[0]


# get_best(costs_v2,acc_test_v2,acc_train_v2)





def demo(costs_main,acc_test,acc_train):
    migtime = [16.6, 16.5, 16.9, 17.1, 17.2, 18.1, 18.2, 18.4, 19.4, 19.3, 20.4, 20.3, 22, 21.7, 22]
    delay = [108, 98, 92, 83, 87, 77, 85, 48, 31, 58, 35, 43, 36, 31, 19]

    fig, ax = plt.subplots()

    plt.xlabel('migration speed (MB/s)')
    plt.ylabel('migration time (s); request delay (ms)')

    """set interval for y label"""
    yticks = range(10, 110, 10)
    ax.set_yticks(yticks)

    """set min and max value for axes"""
    ax.set_ylim([10, 110])
    ax.set_xlim([58, 42])

    x = [57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43]
    plt.plot(x, migtime, "x-", label="migration time")
    plt.plot(x, delay, "+-", label="request delay")

    """open the grid"""
    plt.grid(True)

    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

    plt.show()

# demo()




#car1_v3
# others_file_v3 = '/home/yanggang/longlongaaago/pycharm_cnn_v1/other_para/cnnv3carOtherparameter_0.pk'
#
# other_parameters = load_model(others_file_v3)
# costs_v3 = other_parameters['costs']
# acc_test_v3 = other_parameters['acc_test']
# acc_train_v3 = other_parameters['acc_train']
#
# draw_cost(costs_v3,'v3')
# draw_acc_train(acc_train_v3,'v3')
# draw_acc_test(acc_test_v3,'v3')
# print len(costs_v3)
# print costs_v3
# print acc_train_v3
# print acc_test_v3

##deleteSurplus(costs_v3,acc_test_v3,acc_train_v3,others_file_v3)
#



#car3_v1
# others_file_car3_v1 = '/home/yanggang/longlongaaago/pycharm_cnn_v1_car3/other_para/cnnv1car3therparameter_0.pk'
#
# other_parameters = load_model(others_file_car3_v1)
# costs_car3_v1 = other_parameters['costs']
# acc_test_car3_v1 = other_parameters['acc_test']
# acc_train_car3_v1 = other_parameters['acc_train']
#
# draw_cost(costs_car3_v1,'car3_v1')
# draw_acc_train(acc_train_car3_v1,'car3_v1')
# draw_acc_test(acc_test_car3_v1,'car3_v1')
# print len(costs_car3_v1)
# print costs_car3_v1
# print acc_train_car3_v1
# print acc_test_car3_v1


#car1_v4
others_file_v4 = '/home/yanggang/longlongaaago/pycharm_cnn_v1/other_para/cnnv4carOtherparameter_0.pk'

other_parameters = load_model(others_file_v4)
costs_v4 = other_parameters['costs']
acc_test_v4 = other_parameters['acc_test']
acc_train_v4 = other_parameters['acc_train']

draw_cost(costs_v4,'v4')
draw_acc_train(acc_train_v4,'v4')
draw_acc_test(acc_test_v4,'v4')
print len(costs_v4)
print costs_v4
print acc_train_v4
print acc_test_v4
