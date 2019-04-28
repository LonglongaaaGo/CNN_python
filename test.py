from cnn_fun import *
import time

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

import pickle


# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
plt.show()
# plt.pause(5000)


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


T_train = X_train[0:3,:,:,:]
T_test = Y_test[0:3,:]


# print T_train.shape
# print T_test.shape
#
#
#
# pickle_file = '/home/yanggang/longlongaaago/testpython/DL/cnn/cnn_v1.pk'
#
#
#
#
#
# start = time.time()
# cnn_model_train_test(T_train, T_test,X_test,Y_test,reg=0.001, learning_rate = 0.02, num_epochs = 1000, minibatch_size = 1, print_cost = True)
# end = time.time()
# print end-start





