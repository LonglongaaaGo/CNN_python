#测试的脚本，不是特别重要
from cnn_fun import *
import time

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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


np.random.seed(1)
parameters = initialize_parameters()
X = np.random.randn(2,64,64,3)
Y = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0]])


print '-------train------accuracy-----'
accuracy = test_model(X_train, Y_train, parameters, minibatch_size=1)

