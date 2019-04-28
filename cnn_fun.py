# coding: utf-8

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import math
import scipy
from PIL import Image
from scipy import ndimage
from cnn_utils import *
from dnn_utils_v2 import *
from multiprocessing import Process, Pool
import pickle


np.random.seed(1)


# GRADED FUNCTION: zero_pad
#0填充操作，
#如果为小数，如1.5，则左边填充1，右边填充2，上边填充1，下边填充2
#如果为整数，则四边都填充相同的大小
def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    (m, n_H, n_W, n_C)表示特征图的 ： m = 数量 ， n_H = 高度， n_W = 宽度 ， n_C 维度	
    pad -- float, amount of padding around each image on vertical and horizontal dimensions
    #0填充操作，
    #如果为小数，如1.5，则左边填充1，右边填充2，上边填充1，下边填充2
    #如果为整数，则四边都填充相同的大小
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    ### START CODE HERE ### (≈ 1 line)
    if (pad == 0):
        return X

    if ((pad * 2) % 2 != 0):
        pad_left = int(pad)
        pad_right = pad_left + 1
    else:
        pad_left = int(pad)
        pad_right = int(pad)
    X_pad = np.pad(X, ((0, 0), (pad_left, pad_right), (pad_left, pad_right), (0, 0)), 'constant')
    ### END CODE HERE ###

    return X_pad


# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.
    #一次卷积操作，即特征图和卷积核相同大小维度的一次卷积操作，返回1个值、
    #一次卷积操作为：卷积核移动到某个位置，然后取出和特征图对应维度大小的特征图的局部块，然后进行对应相乘求和，
    #返回1个标量值
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    #a_slice_prev 上1层的特征图的 和卷积核的某个位置的对应局部块
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    # W 卷积核的参数(f, f, n_C_prev)  (卷积核高，卷积核宽，卷积核维度_与上一层特征图通道数对应)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    # b 卷积核的偏置值
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    a scalar value  标量值
    """

    ### START CODE HERE ### (≈ 2 lines of code)
    # Element-wise product between a_slice and W. Do not add the bias yet.
    s = a_slice_prev * W
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + b
    ### END CODE HERE ###

    return Z


# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    单线程的卷积操作
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    #A_prev 上一层所输出的特征图(m, n_H_prev, n_W_prev, n_C_prev)  (数量，高，宽，通道数)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C) 
    #  # W 卷积核的参数(f, f, n_C_prev, n_C)  (卷积核高，卷积核宽，卷积核维度_与上一层特征图通道数对应，卷积核数量)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    #b 卷积核的偏置值 (1, 1, 1, n_C) (1, 1, 1, 卷积核数量)
    hparameters -- python dictionary containing "stride" and "pad"
    #"stride" = 卷积操作的步长
    #"pad"   = 卷积操作的0填充操作值
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    #Z -- 卷积的最终输出, (m, n_H, n_W, n_C)  (数量, 高度, 宽度, 输出维度_和本次卷积核数量挂钩)
    cache -- cache of values needed for the conv_backward() function
    #cache =(A_prev, W, b, hparameters)  (上一层的特征图, 这一层的卷积核参数，这一层的偏置值, 这一层的其他参数)
    #为了保证能够在反向传播的时候直接取值计算相应的梯度
    """

    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i, :, :, :]  # Select ith training example's padded activation

        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over channels (= #filters) of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


# GRADED FUNCTION: pool_forward

def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer
    单线程的下采样操作  
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    #A_prev 上一层所输出的特征图(m, n_H_prev, n_W_prev, n_C_prev)  (数量，高，宽，通道数)

    hparameters -- python dictionary containing "f" and "stride"
    hparameters = "f" 下采样的采样核大小  即 f*f ，"stride"  为相应步长，
    输出的大小其实和卷积操作计算时一样的，只是没有填充（pad）操作。
    mode -- thepad pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    输出 下采样后的特征图  (m, n_H, n_W, n_C)(数量，采样后的高，采样后的宽，通道数)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    
    cache = (A_prev, hparameters)   为了反向梯度求导做准备
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    ### START CODE HERE ###
    for i in range(m):  # loop over the training examples
        for h in range(n_H):  # loop on the vertical axis of the output volume
            for w in range(n_W):  # loop on the horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    ### END CODE HERE ###

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_H, n_W, n_C))
    return A, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    卷积操作的反向传播
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    既然是反向传播，那么计算方式就和卷积的前向传播相反，即"怎么过去的，就怎么回来"，所以输入自然是卷积后的输出层的反向梯度作为输入
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    cache --之前保留下来的卷积核，在这里进行取出计算
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    输出就是对应的 前层的输入特征图，反向传播就像  栈操一样，有点像 前像传播压进去，反向传播弹出计算
    原先在前向传播时的输入，变成了反向传播的输出
    原先在前向传播时的输除，变成了反向传播的输入
    因为计算梯度时，要从后往前，梯度计算回去，也就是链式求导
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    这层卷积核的梯度（f, f, n_C_prev, n_C）对应维度的含义跟前向传播一样：
    有 _prev  后缀的 为前一层的意思
    有  _C  后缀的 为 通道数的意思
    f  代表核的大小  ，可能时卷积核，也可能时 下采样核，因为计算原理时相同的
    之后不再赘述
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    #反向传播的时候，必须要把维度计算回去
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):  # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]
        
        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        # dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        dA_prev[i, :, :, :] = pad_backward(da_prev_pad, pad)
    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    在下采样反向传播时，要用到，这是因为要根据下采样之前的样子，来进行还原，
    下采样之前哪个哪个值最大， 则，给它位置记录下来，在还原的时候，就直接根据采样之前的带下还原维度，然后根据本次计算的最大值的位置，
    把最大值放上去，其他的都填充0
    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """

    ### START CODE HERE ### (≈1 line)
    mask = (x == np.max(x))
    ### END CODE HERE ###

    return mask


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    用的时平局值下采样时的反向传播
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """

    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix (≈1 line)
    average = float(dz) / (n_H * n_W)

    # Create a matrix where every entry is the "average" value (≈1 line)
    a = average * np.ones(shape)
    ### END CODE HERE ###

    return a


def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    下采样时的反向传播，原理同卷积一样，“怎么回去怎么回来”
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
   
    ### START CODE HERE ###

    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(np.shape(A_prev))

    for i in range(m):  # loop over the training examples

        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i, :, :, :]

        for h in range(n_H):  # loop on the vertical axis
            for w in range(n_W):  # loop on the horizontal axis
                for c in range(n_C):  # loop over the channels (depth)

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Compute the backward propagation in both modes.
                    if mode == "max":

                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])


                    elif mode == "average":

                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    ### END CODE ###

    # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev



def softmax_loss_test(W, X, y, b, reg):
    

    #     print '-------------------softmax_loss_test--------------------------------------'
    """
    #模型测试时，使用的 计算softmax 的方法
    #不过这里几个方法冗余了
    
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (c, d) containing weights.
    - X: A numpy array of shape (d, m) containing a minibatch of data.
    - y: A numpy array of shape (m,) containing training labels; y[i] = c means
    - b: A numpy array of shape (c, 1) containing weights.
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    最后全连接：Z = W×X +b  
    然后再套上激活函数 softmax()  ,计算每一个向量的值的大小，不过这里要注意，向量维度的排布
    
    Returns a tuple of:
    - loss as single float
    返回的是loss ->softmax -> dZ，dW，db 的反向梯度的求导的值
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    dZ = np.zeros((W.shape[0], X.shape[1]))

    # da =
    # dAL = np.zeros_like(dZ)

    # 训练数量
    num_train = X.shape[1]
    num_classes = W.shape[0]

    scores = W.dot(X) + b
    Z3 = scores.copy()
    #     print'-----------scores----------'
    #     print scores
    #     print scores.shape

    scores -= np.max(scores, axis=0)[np.newaxis, :]

    exp_scores = np.exp(scores)

    sum_exp_scores = np.sum(exp_scores, axis=0)

    correct_class_score = scores[y, range(num_train)]

    loss = np.sum(np.log(sum_exp_scores)) - np.sum(correct_class_score)

    # 等于最终的预测值
    exp_scores = exp_scores / sum_exp_scores[np.newaxis, :]

    y_hat = exp_scores.copy()

    #     print '--y_hat---'
    #     print y_hat

    dZ = exp_scores.copy()
    for i in xrange(num_train):
        #         print '--exp_scores[:,i][:,np.newaxis] ---'
        #         print exp_scores[:,i][:,np.newaxis].shape
        #         print '---X[:,i][np.newaxis,:]---'
        #         print X[:,i][np.newaxis,:].shape
        #         dW += exp_scores[:,i][:,np.newaxis] *X[:,i][np.newaxis,:]
        dW += np.dot(exp_scores[:, i][:, np.newaxis], X[:, i][np.newaxis, :])
        dW[y[i], :] -= X[:, i]
        dZ[y[i], i] -= 1

    db = (1.0 / num_train) * np.sum(dZ, axis=1).reshape(dZ.shape[0], 1)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    dA_prev = np.dot(W.T, dZ)
    #     print 'dP_2---------------'
    #     print dA_prev.shape

    cache_line_3 = (X, W, b)
    #     print '----softmax----b3-----'
    #     print b.shape
    cache_active_3 = Z3
    #     print '----softmax_loss----y_hat----'
    #     print y_hat

    cache = {
        'cache_line_3': cache_line_3,
        'cache_active_3': cache_active_3,
        'y_hat': y_hat
    }

    grad = {
        'dW3': dW,
        'dZ3': dZ,
        'db3': db,
        'dP_2': dA_prev
    }

    return Z3, loss, grad, cache


def softmax_loss(W, X, y, b, reg):
    """
    最初版本#废弃使用，跳过
    在这里作为参考
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (c, d) containing weights.
    - X: A numpy array of shape (d, m) containing a minibatch of data.
    - y: A numpy array of shape (m,) containing training labels; y[i] = c means
    - b: A numpy array of shape (c, 1) containing weights.
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    dZ = np.zeros((W.shape[0], X.shape[1]))

    # da =
    # dAL = np.zeros_like(dZ)

    # 训练数量
    num_train = X.shape[1]
    num_classes = W.shape[0]

    scores = W.dot(X) + b

    #     print'-----------scores----------'
    #     print scores
    #     print scores.shape

    scores -= np.max(scores, axis=0)[np.newaxis, :]

    exp_scores = np.exp(scores)

    sum_exp_scores = np.sum(exp_scores, axis=0)

    correct_class_score = scores[y, range(num_train)]

    loss = np.sum(np.log(sum_exp_scores)) - np.sum(correct_class_score)

    # 等于最终的预测值
    exp_scores = exp_scores / sum_exp_scores[np.newaxis, :]

    y_hat = exp_scores.copy()

    dZ = exp_scores

    for i in xrange(num_train):
        dW += exp_scores[:, i][:, np.newaxis] * X[:, i][np.newaxis, :]
        dW[y[i], :] -= X[:, i]
        dZ[y[i], i] -= 1

    db = (1.0 / num_train) * np.sum(dZ, axis=1).reshape(dZ.shape[0], 1)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    dA_prev = np.dot(W.T, dZ)
    #     print 'dP_2---------------'
    #     print dA_prev.shape

    cache_line_3 = (X, W, b)
    #     print '----softmax----b3-----'
    #     print b.shape

    cache_active_3 = scores

    cache = {
        'cache_line_3': cache_line_3,
        'cache_active_3': cache_active_3,
        'y_hat': y_hat
    }

    grad = {
        'dW3': dW,
        'dZ3': dZ,
        'db3': db,
        'dP_2': dA_prev
    }

    return scores, loss, grad, cache




# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with cnn. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]

    卷积核参数初始化
    这里写死了，不过原理是一样的
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    np.random.seed(1)

    #取的是正太分布中的随机
    # W1 = np.random.randn(4, 4, 3, 8)
    # W2 = np.random.randn(2, 2, 8, 16)
    W1 = np.random.randn(4, 4, 3, 8) * 0.01
    W2 = np.random.randn(2, 2, 8, 16) * 0.01
    b1 = np.zeros((1, 1, 1, 8))
    b2 = np.zeros((1, 1, 1, 16))

    tag = 0

    parameters = {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2,
        'tag': 0
    }

    return parameters


# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    #在卷积核的最后一层到  全连接层的过度操作
    #实质上时flatten  ，就是把所有特征图拉成一条1向量 
    #保留形状维度参数大小，以便恢复
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[1] * image.shape[2] * image.shape[3], image.shape[0]))
    ### END CODE HERE ###
    shape = (image.shape[0], image.shape[1], image.shape[2], image.shape[3])

    return v, shape


# GRADED FUNCTION: image2vector
def vector2image(vector, img_shape):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    #恢复操作，全连接梯度计算完毕之后，开始反向计算前一层的卷积核的梯度，那么恢复原状以便计算

    (a, b, c, d) = img_shape
    ### START CODE HERE ### (≈ 1 line of code)
    img = vector.reshape((a, b, c, d))
    ### END CODE HERE ###

    return img


# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    #全连接操作
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    ### END CODE HERE ###
  
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def Y_to_y(Y):
    #Y.ndim  是取维度操作，这里要做的就是，如果Y是二维的，那么把它变成1维
    #操作如下对 某1行【0，0，0，0，1，0】  变成   下标 4
    if(Y.ndim>1):
        y = np.where(Y == 1)[1]
        return y
    else:
        return Y


def forward_propagation(X, parameters, Y, reg):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    前向传播，根据前向传播的方法，排布计算即可
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # CONV2D: stride of 1, padding 'SAME'
    hparameters = {
        'stride': 1,
        'pad': 0
    }

    Z1, cache_line_1 = conv_forward_mul_ext(X, W1, b1, hparameters)

    # relu
    A1, cache_activ_1 = relu(Z1)

    # print Z1.shape

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    hparameters = {
        'f': 8,
        'stride': 8
    }

    A_pad_1 = zero_pad(A1, 1.5)

    cache_pad_1 = 1.5
    P1, cache_pool_1 = pool_forward(A_pad_1, hparameters, mode='max')

    # print P1.shape

    cache_1 = {
        'cache_line_1': cache_line_1,
        'cache_activ_1': cache_activ_1,
        'cache_pool_1': cache_pool_1,
        'cache_pad_1': cache_pad_1
    }

    # CONV2D: filters W2, stride 1, padding 'SAME'
    hparameters = {
        'stride': 1,
        'pad': 0
    }
    Z2, cache_line_2 = conv_forward_mul_ext(P1, W2, b2, hparameters)

    # print Z2.shape

    #     # RELU
    A2, cache_activ_2 = relu(Z2)

    #     # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # print A2.shape
    hparameters = {
        'f': 4,
        'stride': 4
    }
    A_pad_2 = zero_pad(A2, 0.5)

    cache_pad_2 = 0.5
    P2, cache_pool_2 = pool_forward(A_pad_2, hparameters, mode='max')

    # print P2.shape

    cache_2 = {
        'cache_line_2': cache_line_2,
        'cache_activ_2': cache_activ_2,
        'cache_pool_2': cache_pool_2,
        'cache_pad_2': cache_pad_2
    }

    #     # FLATTEN
    P2_v, img_shape = image2vector(P2)
    # print P2_v.shape

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"

    if (parameters['tag'] == 0):
        np.random.seed(1)
        W3 = np.random.randn(6, P2_v.shape[0]) * 0.01
        b3 = np.zeros((6, 1))
        parameters['tag'] = 1

        parameters['W3'] = W3
        parameters['b3'] = b3
    else:
        W3 = parameters['W3']
        b3 = parameters['b3']
    #         print '-------else----b3------------'
    #         print b3

    #     print '-----Z1.shape'
    #     print Z1.shape
    #     print '-----A1.shape'
    #     print A1.shape
    #     print '-----P1.shape'
    #     print P1.shape

    #     print '-----Z2.shape'
    #     print Z2.shape
    #     print '-----A2.shape'
    #     print A2.shape
    #     print '-----P2.shape'
    #     print P2.shape

    #     print '-----------W3------------'
    #     print W3.shape
    #     print '-----------P2_v------------'
    #     print P2_v.shape
    #     print '-----------b3------------'
    #     print b3.shape
    #     print b3

    # print W3

    y = Y_to_y(Y)
    scores, loss, grad, cache_3 = softmax_loss_test(W3, P2_v, y, b3, reg)

    Z3 = scores
    print '------Z3------'
    print Z3

    caches = {
        'cache_1': cache_1,
        'cache_2': cache_2,
        'cache_3': cache_3,
        'img_shape': img_shape
    }

    #返回损失值，每一层的梯度，还有前向传播保留下来的参数
    return loss, grad, caches


# GRADED FUNCTION: random_mini_batches

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    保证内存不爆炸，必须要使用minibatch
    Arguments:
    X -- input data, of shape (num,h,w,n_c)
    Y -- true "label" vector (), of shape (num_, number of classes)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    # To make your "random" minibatches the same as ours
    np.random.seed(seed)
    # number of training examples
    m = X.shape[0]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    if(Y.ndim>1):
        shuffled_Y = Y[permutation, :]
    else:
        shuffled_Y = Y[permutation]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m / mini_batch_size)
    num_complete_minibatches = int(num_complete_minibatches)

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :, :, :]
        if(shuffled_Y.ndim>1):
            mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:m, :, :, :]
        if(shuffled_Y.ndim>1):
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:m, :]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:m]
        #         print '-----mini_batch_X-------'
        #         print mini_batch_X.shape
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    #返回的是minibatch的list ，每一个list 都是一个minibatch 
    return mini_batches


def pad_backward(A_prev_pad, pad):

    #pad填充操作的反向传播，其实就是恢复原来大小
    if (pad == 0):
        return A_prev_pad

    if (int(2 * pad) % 2 != 0):
        pad_left = int(pad)
        pad_right = pad_left + 1
    else:
        pad_left = int(pad)
        pad_right = pad_left

    if (A_prev_pad.ndim == 3):
        A_prev = A_prev_pad[pad_left:-pad_right, pad_left:-pad_right, :]
    elif (A_prev_pad.ndim == 4):
        A_prev = A_prev_pad[:, pad_left:-pad_right, pad_left:-pad_right, :]
    
    return A_prev


def cnn_model_backward(grad3, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    反向传播，根据梯度，一步一步往回计算
    Arguments:
    grad = {
        'dW3':dW,
        'dZ3':dZ,
        'db3':db,
        'dP_2':dA_prev
    }
    caches ={
        'cache_1':cache_1,
        'cache_2':cache_2,
        'cache_3':cache_3
    }
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    dW3 = grad3['dW3']
    dZ3 = grad3['dZ3']
    db3 = grad3['db3']

    dPool_2 = grad3['dP_2']

    cache_1 = caches['cache_1']
    cache_2 = caches['cache_2']
    cache_3 = caches['cache_3']
    img_shape = caches['img_shape']

    cache_line_1 = cache_1['cache_line_1']
    cache_activ_1 = cache_1['cache_activ_1']
    cache_pool_1 = cache_1['cache_pool_1']
    cache_pad_1 = cache_1['cache_pad_1']

    cache_line_2 = cache_2['cache_line_2']
    cache_activ_2 = cache_2['cache_activ_2']
    cache_pool_2 = cache_2['cache_pool_2']
    cache_pad_2 = cache_2['cache_pad_2']

    cache_line_3 = cache_3['cache_line_3']
    cache_activ_3 = cache_3['cache_active_3']

    dPool_2 = vector2image(dPool_2, img_shape)

    dPad_2 = pool_backward_mul_ext(dPool_2, cache_pool_2, mode='max')

    dA2 = pad_backward(dPad_2, cache_pad_2)

    dZ2 = relu_backward(dA2, cache_activ_2)

    dPool_1, dW2, db2 = conv_backward_mul_ext(dZ2, cache_line_2)

    dPad_1 = pool_backward_mul_ext(dPool_1, cache_pool_1, mode='max')

    dA1 = pad_backward(dPad_1, cache_pad_1)

    dZ1 = relu_backward(dA1, cache_activ_1)

    dA0, dW1, db1 = conv_backward_mul_ext(dZ1, cache_line_1)

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
        'dW3': dW3,
        'db3': db3,
    }
    #返回梯度
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    梯度更新
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    for i in range(1, 4):
        parameters['W' + str(i)] = parameters['W' + str(i)] - learning_rate * grads['dW' + str(i)]
        #         if(i==3):
        #             grads['db'+str(i)].reshape(parameters['b'+str(i)].shape[0],parameters['b'+str(i)].shape[1] )
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]
    #         print '--------grads[db+str(i)].shape----------'
    #         print '--------parameters['b'+str(i)].shape----------'
    #         print grads['db'+str(i)].shape
    #         print parameters['b'+str(i)].shape
    return parameters


def train_model(X, Y, reg=0.01, learning_rate=0.0075, num_epochs=100, minibatch_size=1, print_cost=True):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape(m, n_H0, n_W0, n_C0)
    Y -- true "label" vector (m,class)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    reg=0.01  正则化参数
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    #梯度保存
    grads = {}
    #损失之保存  
    costs = []  # to keep track of the cost
    (m, n_H0, n_W0, n_C0) = X.shape  # number of examples
    seed = 1
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented

    parameters = initialize_parameters()

    print '-----train----start---'
    the_count = 0
    
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        # number of minibatches of size minibatch_size in the train set
        num_minibatches = int(m / minibatch_size)
        #         print '-----minibatch_cost____init-------'
        #         print minibatch_costtrain_model
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, minibatch_size, seed)
        #         print '----paremeters'
        #         print parameters

        for minibatch in minibatches:
            the_count += 1
            #             start = time.time()
            # Select a minibatchtrain_model
            (minibatch_X, minibatch_Y) = minibatch

            parameters, temp_cost = cnn_batch_model(minibatch_X, minibatch_Y, parameters, reg, learning_rate)

            minibatch_cost += (temp_cost / num_minibatches)

        #             print '-----temp_cost-------'+str(temp_cost) +'------'+str(the_count)

        #             end = time.time()

        #             print '-----time------'
        #             print end-start
        #             print '-----minibatch_cost-------'
        #             print minibatch_cost

        if print_cost and epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per epoch)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters,costs


def cnn_batch_model(X, Y, parameters, reg=0.01, learning_rate=0.0075):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    #cnn_batch_model  1次epoch  包括前向传播，反向传播，以及梯度更新
    Arguments:
    X -- input data, of shape(m, n_H0, n_W0, n_C0)
    Y -- true "label" vector(m,class)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    #     np.random.seed(1)
    #     grads = {}
    #     costs = []                              # to keep track of the cost
    #     m = X.shape[0]                           # number of examples

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented

    # parameters = initialize_parameters()

    #     for i in range(0,num_iterations):

    loss, grad3, caches = forward_propagation_vectorized(X, parameters, Y, reg)

    grads = cnn_model_backward(grad3, caches)
    #     print '-----grads-----'
    #     print grads

    #         print '-----parameters[b3]------'
    #         print parameters['b3']

    # print grads
    parameters = update_parameters(parameters, grads, learning_rate)

    # print loss
    #         if print_cost and i % 1 == 0:
    #             print("Cost after iteration {}: {}".format(i, np.squeeze(loss)))
    #         if print_cost and i % 1 == 0:
    #             costs.append(loss)

    # plot the cost

    #     plt.plot(np.squeeze(costs))
    #     plt.ylabel('cost')
    #     plt.xlabel('iterations (per tens)')
    #     plt.title("Learning rate =" + str(learning_rate))
    #     plt.show()

    return parameters, loss


def test_model_onces(X_test, Y_test, parameters):
    '''
    只用于测试使用
    X_test ,(number,n_h,n_w,n_c)
    Y_test, (classes,number)

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

    return true_num


def Y_hat2y_hat(Y_hat):
    #Y_hat（m，c）
    # m 表示数量， c 表示最终的分类类别
    #直接取最大的位置，进行比较
    Y_hat = np.where(Y_hat == np.max(Y_hat, axis=0))[0]

    #     print '----Y_hat------'
    #     print Y_hat
    Y_hat = np.array(Y_hat)
    # Y_hat = Y_hat.T
    y_hat = np.squeeze(Y_hat)

    return y_hat


def cnn_model_train_test(X_train, Y_train, X_test, Y_test, reg=0.01, learning_rate=0.01, num_epochs=40,
                         minibatch_size=2, print_cost=True):
    '''
    X_train  训练样本(m, n_H0, n_W0, n_C0)
    Y_train  训练样本 的标签(classes,number)
    X_test   测试样本(m, n_H0, n_W0, n_C0)
    Y_test   测试样本的标签(classes,number)
    reg  正则化值 
    learning_rate （学习率）
    num_epochs  epoch 的总数
    minibatch_size  每次训练  拿多少样本
    print_cost   是否打印
    '''

    
    parameters = train_model(X_train, Y_train, reg, learning_rate, num_epochs, minibatch_size, print_cost)

    print '-------train------accuracy-----'
    accuracy = test_model(X_train, Y_train, parameters, minibatch_size)

    print '-------test------accuracy-----'
    accuracy = test_model(X_test, Y_test, parameters, minibatch_size)


def test_model(X_test, Y_test, parameters, minibatch_size=1):
    '''
    X_test ,(number,n_h,n_w,n_c)
    Y_test, (classes,number)
    只用于测试准确率

    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
   
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_accur -- True to print the cost every 100 epochs

    Returns:
    accuracy --
    """

    '''
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
        acc_temp = test_model_onces(minibatch_X, minibatch_Y, parameters)
        acc_num += acc_temp
        temp_ = minibatch_X.shape[0]

        print '---temp_acc-----'
        print '--' + str(float(acc_temp) / temp_)

    accuracy = float(acc_num) / m

    print '---accuracy--'
    print accuracy

    return accuracy


def sub_con_forward(i, a_prev_pad, parameter):
    (n_H, n_W, n_C, W, b, hparameters) = parameter
    #sub_con_forward  给予多进程的实现，由进程池来对每一个卷积进行分别计算，最后整合
    Z = np.zeros((n_H, n_W, n_C))

    #     print '---sub_con_forward---'+str(i)

    stride = hparameters['stride']
    f = W.shape[0]
    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            for c in range(n_C):  # loop over channels (= #filters) of the output volume

                # Find the corners of the current "slice" (≈4 lines)
                vert_start = stride * h
                vert_end = vert_start + f
                horiz_start = stride * w
                horiz_end = horiz_start + f

                # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                Z[h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    slice_Z = (i, Z)

    return slice_Z


# GRADED FUNCTION: conv_forward

def conv_forward_mul(A_prev, W, b, hparameters):
    #此方法以废弃，普通的进程池不能随主进程关闭而关闭，禁止使用！！！！
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    pool = Pool()
    pool.daemon = True
    # pool.daemon = True
    parameter = (n_H, n_W, n_C, W, b, hparameters)

    res_l = []

    for i in range(m):  # loop over the batch of training examples

        a_prev_pad = A_prev_pad[i, :, :, :]  # Select ith training example's padded activation
        # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        res = pool.apply_async(sub_con_forward, (i, a_prev_pad, parameter,))
        res_l.append(res)

    try:
        pool.close()  # 关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成
        pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    except KeyboardInterrupt:

        pool.terminate()  # 直接关闭进程池
        #         pool.close() #关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成
        pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    for res in res_l:
        i, slice_Z = res.get()
        Z[i, :, :, :] = slice_Z
    #         print '------i----'+str(i)
    #         print slice_Z

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


def conv_backward_mul(dZ, cache):
    
    #同上！！！禁止使用，不带管理的进程池，造成程序无法关闭

    """

    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    pool = Pool()
    pool.daemon = True
    parameter = (dW, db, n_H_prev, n_W_prev, n_C_prev, n_H, n_W, n_C, W, b, hparameters)
    res_l = []

    for i in range(m):  # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        res = pool.apply_async(sub_con_backward, (i, dZ[i, :, :, :], a_prev_pad, da_prev_pad, parameter,))
        res_l.append(res)

    pool.close()  # 关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    for res in res_l:
        #         print (res.get())
        i, dA_prev_sub, dW_sub, db_sub = res.get()
        dA_prev[i, :, :, :] = dA_prev_sub
        dW += dW_sub
        db += db_sub

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


def sub_con_backward(i, dZ_, a_prev_pad, da_prev_pad, parameter):
    
    '''
    基于多进程的卷积反向传播，原理同单线程的一样，只是把每一个特征图都交给对应的进程池管理者来管理
    '''
    (dW, db, n_H_prev, n_W_prev, n_C_prev, n_H, n_W, n_C, W, b, hparameters) = parameter

    dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))

    #     print '--sub_con_backward--'
    #     print dA_prev.shape

    stride = hparameters['stride']
    pad = hparameters['pad']
    f = W.shape[0]

    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            for c in range(n_C):  # loop over the channels of the output volume
                # Find the corners of the current "slice"
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f

                # Use the corners to define the slice from a_prev_pad
                a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                # Update gradients for the window and the filter's parameters using the code formulas given above
                #                 da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                #                 dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                #                 db[:,:,:,c] += dZ[i, h, w, c]

                da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ_[h, w, c]
                dW[:, :, :, c] += a_slice * dZ_[h, w, c]
                db[:, :, :, c] += dZ_[h, w, c]

    # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
    dA_prev[:, :, :] = pad_backward(da_prev_pad, pad)

    #     print dA_prev

    slice_dA_prev = (i, dA_prev, dW, db)

    return slice_dA_prev


def pool_backward_mul(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer
    ###禁止使用，进程池没有管理者，程序无法正常退出
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    ### START CODE HERE ###

    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(np.shape(A_prev))

    pool = Pool()
    pool.daemon = True
    parameter = (n_H_prev, n_W_prev, n_C_prev, n_H, n_W, n_C, hparameters)

    res_l = []

    for i in range(m):  # loop over the training examples

        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i, :, :, :]
        # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        res = pool.apply_async(sub_pool_backward, (i, parameter, mode, a_prev, dA[i, :, :, :],))

        res_l.append(res)

    pool.close()  # 关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    for res in res_l:
        i, dA_prev_sub = res.get()
        dA_prev[i, :, :, :] = dA_prev_sub

    ### END CODE ###

    # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


def sub_pool_backward(i, parameter, mode, a_prev, dA_sub):
    
    ##下采样的反向传播多进程 实现的一个自操作，将每一个下采样的方向传播交给多进程管理者
    (n_H_prev, n_W_prev, n_C_prev, n_H, n_W, n_C, hparameters) = parameter
    stride = hparameters['stride']
    f = hparameters['f']
    
    dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))

    for h in range(n_H):  # loop on the vertical axis
        for w in range(n_W):  # loop on the horizontal axis
            for c in range(n_C):  # loop over the channels (depth)

                # Find the corners of the current "slice" (≈4 lines)
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f

                # Compute the backward propagation in both modes.
                if mode == "max":
                    # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    # Create the mask from a_prev_slice (≈1 line)
                    mask = create_mask_from_window(a_prev_slice)
                    # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                    dA_prev[vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA_sub[h, w, c])


                elif mode == "average":

                    # Get the value a from dA (≈1 line)
                    da = dA_sub[h, w, c]
                    # Define the shape of the filter as fxf (≈1 line)
                    shape = (f, f)
                    # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                    dA_prev[vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

    slice_dA_prev = (i, dA_prev)

    return slice_dA_prev





# GRADED FUNCTION: conv_forward

def conv_forward_mul_ext(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    ###ProcessPoolExecutor  多进程的实现，把每一个卷积都交给了进程池管理者，原理同单进程
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
    n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    pool = ProcessPoolExecutor()
    #     pool = Pool()
    #     pool.daemon = True
    # pool.daemon = True
    parameter = (n_H, n_W, n_C, W, b, hparameters)

    res_l = []
    #     with ProcessPoolExecutor() as executor: 
    for i in range(m):  # loop over the batch of training examples

        a_prev_pad = A_prev_pad[i, :, :, :]

        futures = pool.submit(sub_con_forward, i, a_prev_pad, parameter)
        res_l.append(futures)

    pool.shutdown(wait=True)

    for res in res_l:
        i, slice_Z = res.result()
        Z[i, :, :, :] = slice_Z
    #         print '------i----'+str(i)
    #         print slice_Z

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


def conv_backward_mul_ext(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    ###ProcessPoolExecutor  多进程的实现，把每一个卷积反向传播的操作都交给了进程池管理者，原理同单进程
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    #     print 'dA_prev'
    #     print dA_prev.shape
    #     print 'dW'
    #     print dW.shape
    #     print 'db'
    #     print db.shape

    #     print 'dZ'
    #     print dZ.shape

    pool = ProcessPoolExecutor()
    #     pool.daemon = True
    parameter = (dW, db, n_H_prev, n_W_prev, n_C_prev, n_H, n_W, n_C, W, b, hparameters)
    res_l = []

    for i in range(m):  # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        futures = pool.submit(sub_con_backward, i, dZ[i, :, :, :], a_prev_pad, da_prev_pad, parameter)

        #         #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        #         res = pool.apply_async(sub_con_backward, (i,dZ[i,:,:,:],a_prev_pad,da_prev_pad,parameter, ))   
        res_l.append(futures)

    #     pool.close() #关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成
    #     pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束  

    pool.shutdown(wait=True)
    for res in res_l:
        #         print (res.get())
        i, dA_prev_sub, dW_sub, db_sub = res.result()
        dA_prev[i, :, :, :] = dA_prev_sub
        dW += dW_sub
        db += db_sub

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


def pool_backward_mul_ext(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer
    ###ProcessPoolExecutor  多进程的实现，把每一个下采样反向传播的操作都交给了进程池管理者，原理同单进程
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    ### START CODE HERE ###

    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache

    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(np.shape(A_prev))

    pool = ProcessPoolExecutor()

    #     pool = Pool()
    #     pool.daemon = True
    parameter = (n_H_prev, n_W_prev, n_C_prev, n_H, n_W, n_C, hparameters)

    res_l = []

    for i in range(m):  # loop over the training examples

        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i, :, :, :]

        futures = pool.submit(sub_pool_backward, i, parameter, mode, a_prev, dA[i, :, :, :])
        #         #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        #         res = pool.apply_async(sub_pool_backward, (i,parameter,mode,a_prev,dA[i,:,:,:], ))   

        res_l.append(futures)

    #     pool.close() #关闭进程池，防止进一步操作。如果所有操作持续挂起，它们将在工作进程终止前完成
    #     pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

    pool.shutdown(wait=True)

    for res in res_l:
        i, dA_prev_sub = res.result()
        dA_prev[i, :, :, :] = dA_prev_sub

    ### END CODE ###

    # Making sure your output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


def load_model(pickle_file):
    #读取之前保存的pickle 文件（实质上时键值对数据序列化文件）
    #pickle_file  文件全路径
    with open(pickle_file, 'r') as f:
        data = pickle.load(f)
    return data


def save_model(data, pickle_file):
    #保存的pickle 文件（实质上时键值对数据序列化文件）
    #data，字典文件
    #pickle_file  要保存的文件全路径，包括文件名称
    with open(pickle_file, 'w') as f:
        pickle.dump(data, f)

def save_model_byspeed(data, pickle_file,num):
    #data，字典文件  每一次epoch 之后的  卷积核参数
    #pickle_file  要保存的文件全路径，包括文件名称
    #num 后缀
    temp = pickle_file.split('_')
    temp_num = temp[-1].split('.')
    _num = temp_num[0]
    print _num
    type = temp_num[1]
    print type
    _num = int(_num) + num
    final_temp = ''
    del temp[-1]
    for i in temp:
        final_temp = final_temp+i+'_'
    pickle_file = final_temp + str(_num)+'.'+type
    print pickle_file
    with open(pickle_file, 'w') as f:
        pickle.dump(data, f)



def cnn_model_train_test_pk(other_parameters,parameters ,X_train, Y_train, X_test, Y_test, reg=0.01, learning_rate=0.01, num_epochs=40,
                            minibatch_size=2, print_cost=True):

    '''
    保存了['costs']、['acc_test']、['acc_train'] 训练过程的数据
    保存了训练过程的参数数据
    '''
    costs_main = other_parameters['costs']
    acc_test = other_parameters['acc_test']
    acc_train = other_parameters['acc_train']


    parameters,costs = train_model_pk(parameters, X_train, Y_train, reg, learning_rate, num_epochs, minibatch_size, print_cost)


    print '-------train------accuracy-----'
    accuracy_train = test_model(X_train, Y_train, parameters, minibatch_size)
    print '-------train------accuracy-----'+str(accuracy_train)

    print '-------test------accuracy-----'
    accuracy_test = test_model(X_test, Y_test, parameters, minibatch_size)
    print '-----------test------accuracy-----' + str(accuracy_test)

    costs_main.extend(costs)
    acc_test.append(accuracy_test)
    acc_train.append(accuracy_train)

    other_parameters['costs'] = costs_main
    other_parameters['acc_test'] = acc_test
    other_parameters['acc_train'] = acc_train

    return parameters,other_parameters



def train_model_pk(parameters, X, Y, reg=0.01, learning_rate=0.0075, num_epochs=100, minibatch_size=1, print_cost=True):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
     '''
    X_train  训练样本(m, n_H0, n_W0, n_C0)
    Y_train  训练样本 的标签(classes,number)
  
    learning_rate （学习率）
    num_epochs  epoch 的总数
    minibatch_size  每次训练  拿多少样本
    print_cost   是否打印
    '''
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)

    costs = []  # to keep track of the cost
    (m, n_H0, n_W0, n_C0) = X.shape  # number of examples
    seed = 1
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented

    print parameters

    print '-----train----start---'
    the_count = 0
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        # number of minibatches of size minibatch_size in the train set
        num_minibatches = int(m / minibatch_size)
        #         print '-----minibatch_cost____init-------'
        #         print minibatch_costtrain_model
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, minibatch_size, seed)
        #         print '----paremeters'
        #         print parameters

        for minibatch in minibatches:
            the_count += 1
            #             start = time.time()
            # Select a minibatchtrain_model
            (minibatch_X, minibatch_Y) = minibatch

            parameters, temp_cost = cnn_batch_model(minibatch_X, minibatch_Y, parameters, reg, learning_rate)

            minibatch_cost += (temp_cost / num_minibatches)

            # print '-----temp_cost-------'+str(temp_cost) +'------'+str(the_count)

        #             end = time.time()

        #             print '-----time------'
        #             print end-start
        #             print '-----minibatch_cost-------'
        #             print minibatch_cost

        if print_cost and epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)

    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per epoch)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters,costs


# GRADED FUNCTION: initialize_parameters

def initialize_parameters_pk(pkfile, load=False):
    """
    Initializes weight parameters to build a neural network with cnn. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    从文件里读取参数，若没有则随机生成
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    np.random.seed(1)

    W1 = np.random.randn(4, 4, 3, 8) * 0.01
    W2 = np.random.randn(2, 2, 8, 16) * 0.01
    b1 = np.zeros((1, 1, 1, 8))
    b2 = np.zeros((1, 1, 1, 16))

    tag = 0

    parameters = {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2,
        'tag': 0
    }

    if (load):
        try:
            parameters = load_model(pkfile)
        except IOError:
            print 'no file'

        except EOFError:
            print 'file_zero'

    return parameters


#
# def save_model(data, pickle_file):
#     with open(pickle_file, 'w') as f:
#         pickle.dump(data, f)


def load_parameter(pickle_file):
    #读取参数
    with open(pickle_file, 'r') as f:
        data = pickle.load(f)
    return data



def run_cnn_file(hyperparameter_file):
    #未使用，本来打算将参数全部放在一个字典文件中，然后每次读取超参数
    try:
        hyperparameter = load_parameter(hyperparameter_file)
    except IOError:
        print 'plase save your parameterfile'
        return
    except EOFError:
        print 'plase save your parameterfile'
        return

    continue_save = hyperparameter['continue_save']

    pkfile = hyperparameter['pkfile']
    load = hyperparameter['load']
    save = hyperparameter['save']
    X_train =hyperparameter['X_train']
    Y_train = hyperparameter['Y_train']
    X_test = hyperparameter['X_test']
    Y_test = hyperparameter['Y_test']
    reg = hyperparameter['reg']
    learning_rate = hyperparameter['learning_rate']
    num_epochs = hyperparameter ['num_epochs']
    minibatch_size = hyperparameter['minibatch_size']
    print_cost = hyperparameter['print_cost']



    cnn_model_train_test_pk_reload(pkfile, load,save,continue_save, X_train, Y_train, X_test, Y_test, reg, learning_rate,
                            num_epochs, minibatch_size, print_cost)



def cnn_model_train_test_pk_reload(pkfile,load ,save,continue_save,X_train, Y_train, X_test, Y_test, reg=0.01, learning_rate=0.01, num_epochs=40,
                            minibatch_size=2, print_cost=True):
    #未使用，跳过

    parameters = train_model_pk_reload(pkfile,load, X_train, Y_train, reg, learning_rate, num_epochs, minibatch_size, print_cost)

    save_model_reload(parameters, pkfile)

    print '-------train------accuracy-----'
    accuracy = test_model(X_train, Y_train, parameters, minibatch_size)

    print '-------test------accuracy-----'
    accuracy = test_model(X_test, Y_test, parameters, minibatch_size)



def train_model_pk_reload(pkfile,load,save,continue_save,iter, X, Y, reg=0.01, learning_rate=0.0075, num_epochs=100, minibatch_size=1, print_cost=True):
    #未使用，跳过    

    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    (m, n_H0, n_W0, n_C0) = X.shape  # number of examples
    seed = 1
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented

    parameters = initialize_parameters_pk(pkfile, load)

    print parameters

    print '-----train----start---'
    the_count = 0
    for epoch in range(num_epochs):
        minibatch_cost = 0.
        # number of minibatches of size minibatch_size in the train set
        num_minibatches = int(m / minibatch_size)
        #         print '-----minibatch_cost____init-------'
        #         print minibatch_costtrain_model
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, minibatch_size, seed)
        #         print '----paremeters'
        #         print parameters

        for minibatch in minibatches:
            the_count += 1
            #             start = time.time()
            # Select a minibatchtrain_model
            (minibatch_X, minibatch_Y) = minibatch

            parameters, temp_cost = cnn_batch_model(minibatch_X, minibatch_Y, parameters, reg, learning_rate)

            minibatch_cost += (temp_cost / num_minibatches)

        #             print '-----temp_cost-------'+str(temp_cost) +'------'+str(the_count)

        #             end = time.time()

        #             print '-----time------'
        #             print end-start
        #             print '-----minibatch_cost-------'
        #             print minibatch_cost

        if print_cost and epoch % 1 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)
        if(epoch % iter == 0):
            save_model_reload(parameters,pkfile,save,continue_save,iter)



    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters





def save_model_reload(data, pickle_file,save,continue_save,iter):
   #未使用，跳过 
   if(save):
        if(continue_save):
            temp = pickle_file.split('_')
            name = temp[-1]
            num_temp = name.split('.')
            num = num_temp[-2]
            num = iter+int(num)
            pickle_file = temp +'_'+str(num)+'.'+num_temp[-1]

            with open(pickle_file, 'w') as f:
                pickle.dump(data, f)
        else :

            with open(pickle_file, 'w') as f:
                pickle.dump(data, f)

def load_model_reload(pickle_file):
    with open(pickle_file, 'r') as f:
        data = pickle.load(f)
    return data



#------------------------------vectorized--------------------------------------

def softmax_loss_vectorized(W, X, y, b, reg):
    # print '--=------softmax_loss_vectorized----------------------------------------------'

    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (m, D) containing a minibatch of data.
    m 表示样本数量，c 标识类别   ，D 标识输入  神经元数量
    - y: A numpy array of shape (m,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    dZ = np.zeros((X.shape[0], W.shape[1]))

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    # num_classes = W.shape[1]

    scores = X.dot(W) + b
    #     print 'scores'
    #     print scores
    Z3 = scores.copy()
    # print 'Z3'
    # print Z3.shape

    scores -= np.max(scores, axis=1)[:, np.newaxis]

    #     print 'scores'
    #     print scores

    exp_scores = np.exp(scores)
    #     print 'exp_scores'
    #     print exp_scores

    sum_exp_scores = np.sum(exp_scores, axis=1)
    #     print 'sum_exp_scores'
    #     print sum_exp_scores

    correct_class_score = scores[range(num_train), y]

    #     print 'correct_class_score'
    #     print correct_class_score

    loss = np.sum(np.log(sum_exp_scores)) - np.sum(correct_class_score)
    #     print 'loss'
    #     print loss

    exp_scores = exp_scores / sum_exp_scores[:, np.newaxis]

    y_hat = exp_scores.copy()
    # print 'y_hat'
    # print y_hat

    dZ = exp_scores.copy()

    # maybe here can be rewroten into matrix operations
    for i in xrange(num_train):
        dW += exp_scores[i] * X[i][:, np.newaxis]
        dW[:, y[i]] -= X[i]
        dZ[i, y[i]] -= 1

    db = (1.0 / num_train) * np.sum(dZ, axis=0)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    dA_prev = np.dot(dZ, W.T)
    #     print 'dP_2---------------'
    #     print dA_prev.shape

    cache_line_3 = (X, W, b)
    #     print '----softmax----b3-----'
    #     print b.shape
    cache_active_3 = Z3
    #     print '----softmax_loss----y_hat----'
    #     print y_hat

    cache = {
        'cache_line_3': cache_line_3,
        'cache_active_3': cache_active_3,
        'y_hat': y_hat
    }

    grad = {
        'dW3': dW,
        'dZ3': dZ,
        'db3': db,
        'dP_2': dA_prev
    }

    return scores, loss, grad, cache


def forward_propagation_vectorized(X, parameters, Y, reg):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    _vectorized  后缀的方法实质上是 维度的变化，即可能进行了转置操作，实质上时一样的，前向传播
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # print 'W1'
    # print W1
    # print 'b1'
    # print b1

    # CONV2D: stride of 1, padding 'SAME'
    hparameters = {
        'stride': 1,
        'pad': 1.5
    }

    Z1, cache_line_1 = conv_forward_mul_ext(X, W1, b1, hparameters)

    # relu
    A1, cache_activ_1 = relu(Z1)

    # print '--'


    # print Z1.shape

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    hparameters = {
        'f': 8,
        'stride': 8
    }

    A_pad_1 = zero_pad(A1, 0)

    cache_pad_1 = 0
    P1, cache_pool_1 = pool_forward(A_pad_1, hparameters, mode='max')

    # print P1.shape

    cache_1 = {
        'cache_line_1': cache_line_1,
        'cache_activ_1': cache_activ_1,
        'cache_pool_1': cache_pool_1,
        'cache_pad_1': cache_pad_1
    }

    # CONV2D: filters W2, stride 1, padding 'SAME'
    hparameters = {
        'stride': 1,
        'pad': 0.5
    }
    Z2, cache_line_2 = conv_forward_mul_ext(P1, W2, b2, hparameters)

    # print Z2.shape

    #     # RELU
    A2, cache_activ_2 = relu(Z2)

    #     # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # print A2.shape
    hparameters = {
        'f': 4,
        'stride': 4
    }
    A_pad_2 = zero_pad(A2, 0)

    cache_pad_2 = 0
    P2, cache_pool_2 = pool_forward(A_pad_2, hparameters, mode='max')

    # print P2.shape

    cache_2 = {
        'cache_line_2': cache_line_2,
        'cache_activ_2': cache_activ_2,
        'cache_pool_2': cache_pool_2,
        'cache_pad_2': cache_pad_2
    }

    #     # FLATTEN
    P2_v, img_shape = image2vector_ver(P2)
    # print P2_v.shape

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"

    if (parameters['tag'] == 0):
        np.random.seed(1)
        W3 = np.random.randn(P2_v.shape[1], 6) * 0.01
        b3 = np.zeros((1, 6))
        parameters['tag'] = 1

        parameters['W3'] = W3
        parameters['b3'] = b3


    W3 = parameters['W3']
    b3 = parameters['b3']
    #         print '-------else----b3------------'
    #         print b3

        # print '-----Z1.shape'
        # print Z1.shape
        # print '-----A1.shape'
        # print A1.shape
        # print '-----P1.shape'
        # print P1.shape
        #
        # print '-----Z2.shape'
        # print Z2.shape
        # print '-----A2.shape'
        # print A2.shape
        # print '-----P2.shape'
        # print P2.shape
        #
        # print '-----------W3------------'
        # print W3.shape
        # print '-----------P2_v------------'
        # print P2_v.shape
        # print '-----------b3------------'
        # print b3.shape
        # print b3

    # print W3

    y = Y_to_y(Y)

    # print '-=------W3-----------'
    # print W3.shape
    # print '--------P2_V--------'
    # print P2_v.shape
    # print '-------y----------'
    # print y.shape
    # print '--------b3--------'
    # print b3.shape

    scores, loss, grad, cache_3 = softmax_loss_vectorized(W3, P2_v, y, b3, reg)

    Z3 = scores
    # print Z3.shape

    caches = {
        'cache_1': cache_1,
        'cache_2': cache_2,
        'cache_3': cache_3,
        'img_shape': img_shape
    }

    return loss, grad, caches


# GRADED FUNCTION: image2vector
def image2vector_ver(image):
    """
    Argument:
    flatten  操作
    卷积层最后一层到 全连接层时，要做flatten
    image -- a numpy array of shape (m,length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[0], image.shape[1] * image.shape[2] * image.shape[3]))
    ### END CODE HERE ###
    shape = (image.shape[0], image.shape[1], image.shape[2], image.shape[3])

    return v, shape

def Y_hat2y_hat_ver(X):
    #X（number，class）
    #[0,0.3,0.5,0.2,0,0]
    #[0.8,0,0,0.2,0,0]
    #变成 [2,0]
    yhat = np.zeros((X.shape[0],))
    for i in xrange(X.shape[0]):
        yhat[i] = np.where(X[i]==np.max(X[i]))[0]
    yhat = yhat.astype(np.int32)
    return yhat


def softmax_forward(W, X,b):
    # print '--=------softmax_loss_vectorized----------------------------------------------'
    #softmax_forward前向传播
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (m, D) containing a minibatch of data.
    - y: A numpy array of shape (m,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.

    num_train = X.shape[0]
    # num_classes = W.shape[1]

    scores = X.dot(W) + b
    #     print 'scores'
    #     print scores

    scores -= np.max(scores, axis=1)[:, np.newaxis]

    #     print 'scores'
    #     print scores

    exp_scores = np.exp(scores)
    #     print 'exp_scores'
    #     print exp_scores

    sum_exp_scores = np.sum(exp_scores, axis=1)
    #     print 'sum_exp_scores'
    #     print sum_exp_scores

    exp_scores = exp_scores / sum_exp_scores[:, np.newaxis]

    y_hat = exp_scores.copy()
    # print 'y_hat'
    # print y_hat
    cache = {
        'y_hat': y_hat
    }

    return cache


def forward_propagation_test(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    #只进行前向传播，用于测试
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # print 'W1'
    # print W1
    # print 'b1'
    # print b1

    # CONV2D: stride of 1, padding 'SAME'
    hparameters = {
        'stride': 1,
        'pad': 1.5
    }

    Z1, cache_line_1 = conv_forward_mul_ext(X, W1, b1, hparameters)

    # relu
    A1, cache_activ_1 = relu(Z1)

    # print '--'


    # print Z1.shape

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    hparameters = {
        'f': 8,
        'stride': 8
    }

    A_pad_1 = zero_pad(A1, 0)

    cache_pad_1 = 0
    P1, cache_pool_1 = pool_forward(A_pad_1, hparameters, mode='max')

    # print P1.shape

    # cache_1 = {
    #     'cache_line_1': cache_line_1,
    #     'cache_activ_1': cache_activ_1,
    #     'cache_pool_1': cache_pool_1,
    #     'cache_pad_1': cache_pad_1
    # }

    # CONV2D: filters W2, stride 1, padding 'SAME'
    hparameters = {
        'stride': 1,
        'pad': 0.5
    }
    Z2, cache_line_2 = conv_forward_mul_ext(P1, W2, b2, hparameters)

    # print Z2.shape

    #     # RELU
    A2, cache_activ_2 = relu(Z2)

    #     # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # print A2.shape
    hparameters = {
        'f': 4,
        'stride': 4
    }
    A_pad_2 = zero_pad(A2, 0)

    cache_pad_2 = 0
    P2, cache_pool_2 = pool_forward(A_pad_2, hparameters, mode='max')

    # print P2.shape

    # cache_2 = {
    #     'cache_line_2': cache_line_2,
    #     'cache_activ_2': cache_activ_2,
    #     'cache_pool_2': cache_pool_2,
    #     'cache_pad_2': cache_pad_2
    # }

    #     # FLATTEN
    P2_v, img_shape = image2vector_ver(P2)
    # print P2_v.shape

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"

    if (parameters['tag'] == 0):
        np.random.seed(1)
        W3 = np.random.randn(P2_v.shape[1], 6) * 0.01
        b3 = np.zeros((1, 6))
        parameters['tag'] = 1

        parameters['W3'] = W3
        parameters['b3'] = b3

    W3 = parameters['W3']
    b3 = parameters['b3']
    #         print '-------else----b3------------'
    #         print b3

        # print '-----Z1.shape'
        # print Z1.shape
        # print '-----A1.shape'
        # print A1.shape
        # print '-----P1.shape'
        # print P1.shape
        #
        # print '-----Z2.shape'
        # print Z2.shape
        # print '-----A2.shape'
        # print A2.shape
        # print '-----P2.shape'
        # print P2.shape
        #
        # print '-----------W3------------'
        # print W3.shape
        # print '-----------P2_v------------'
        # print P2_v.shape
        # print '-----------b3------------'
        # print b3.shape
        # print b3

    # print W3

    # y = Y_to_y(Y)

    # print '-=------W3-----------'
    # print W3.shape
    # print '--------P2_V--------'
    # print P2_v.shape
    # print '-------y----------'
    # print y.shape
    # print '--------b3--------'
    # print b3.shape

    cache = softmax_forward(W3, P2_v, b3)

    # Z3 = scores
    # print Z3.shape

    # caches = {
    #     'cache_1': cache_1,
    #     'cache_2': cache_2,
    #     'cache_3': cache_3,
    #     'img_shape': img_shape
    # }

    return cache


def test_model_onces_forward(X_test, parameters):
    '''
    X_test ,(number,n_h,n_w,n_c)
    Y_test, (classes,number)
    只进行前向传播 1次
    return :
    the true number of the X_test
    '''

    cache = forward_propagation_test(X_test, parameters)
    AL = cache['y_hat']

    print '-----AL------'
    print AL
    # y_hat = Y_hat2y_hat(AL)
    y_hat = Y_hat2y_hat_ver(AL)

    print'------y_hat-----'
    print y_hat

    #     accuracy = float(true)/len(reult)

    #     print reult

    #     print '---accuracy--'
    #     print accuracy

    return y_hat


#-----main---------------------------
def train_main(others_file,pkfile, load,X_train, Y_train,X_test,Y_test,main_epoch=5,reg=0.0, learning_rate = 0.005, num_epochs = 1, minibatch_size = 2, print_cost = True):
    '''
    主要方法：
    others_file，每次epoch 训练之后的准确率和损失值等 信息 文件的读取，若没有则重新创建
    pkfile ， 上次训练后的 卷积核参数 ，从中读取，若没有，则重新创建
    load ，是否读取上次的模型参数，并继续训练
    X_train  训练样本(m, n_H0, n_W0, n_C0)
    Y_train  训练样本 的标签(classes,number)
    X_test   测试样本(m, n_H0, n_W0, n_C0)
    Y_test   测试样本的标签(classes,number)
    main_epoch  epoch 的总数
    reg  正则化值 
    learning_rate （学习率）
    num_epochs  设成1，必须设成1，多套了个循环。
    minibatch_size  每次训练  拿多少样本
    print_cost   是否打印
    '''
    other_parameters =  load_other_para(others_file)

    print other_parameters

    parameters = initialize_parameters_pk(pkfile, load)

    for i in range(main_epoch):
        parameters ,other_parameters=cnn_model_train_test_pk(other_parameters,parameters,X_train, Y_train,X_test,Y_test,reg, learning_rate , num_epochs, minibatch_size , print_cost )

        save_model_byspeed(parameters, pkfile,(i+1)*num_epochs)
        save_model(other_parameters,others_file)
        print str(i)+'is ok'


def load_other_para(others_file):
    #读取每次epoch 训练之后的准确率和损失值 等
    costs = []
    acc_test =[]
    acc_train =[]
    others = {
        'costs': costs,
        'acc_test': acc_test,
        'acc_train': acc_train
    }

    try:
        others = load_model(others_file)
    except IOError:
        print 'no file'

    except EOFError:
        print 'file_zero'

    return others
