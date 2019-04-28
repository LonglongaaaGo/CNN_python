
#未使用  ，仅做参考参考
from cnn_fun import *


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

pkfile = '/home/longlongaaago/WorkSpace/pycharm/pycharm_cnn_v1/cnn_v1.pk'

#print the cost
print_cost = True
#load the early parameter
load = True
#save the new parameter
save = True

reg = 0.01
learning_rate = 0.02
num_epochs = 10
minibatch_size = 1


hyperparameter={

    'pkfile':pkfile,
    'load':load,
    'save':save,
    'X_train':X_train,
    'Y_train':Y_train,
    'X_test':X_test,
    'Y_test':Y_test,
    'reg':reg,
    'learning_rate':learning_rate,
    'num_epochs':num_epochs,
    'minibatch_size':minibatch_size,
    'print_cost':print_cost

}
