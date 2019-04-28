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


# GRADED FUNCTION: image2vector
def image2vector_ver(image):
    """
    Argument:
    image -- a numpy array of shape (m,length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[0], image.shape[1] * image.shape[2] * image.shape[3]))
    ### END CODE HERE ###
    shape = (image.shape[0], image.shape[1], image.shape[2], image.shape[3])

    return v, shape



# GRADED FUNCTION: conv_single_step

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
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


# GRADED FUNCTION: zero_pad

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

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
    print pad_left
    print pad_right

    X_pad = np.pad(X, ((0, 0), (pad_left, pad_right), (pad_left, pad_right), (0, 0)), 'constant')
    ### END CODE HERE ###

    return X_pad


# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
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
    # mode = hparameters['same']

    # if(mode =='same'):
    #     # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    #     n_H = ceil(n_H_prev / stride)
    #     n_W = ceil(n_W_prev / stride)
    # else:
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

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- thepad pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # mode = hparameters['same']

    # if (mode == 'same'):
    #     # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    #     n_H = ceil(n_H_prev / stride)
    #     n_W = ceil(n_W_prev / stride)
    # else:
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume

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
    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache



def conv_backward(dZ, cache):
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

    for i in range(m):                       # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]

        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        #dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        dA_prev[i, :, :, :] = pad_backward(da_prev_pad,pad)
    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db

def softmax_loss_vectorized(W, X, y, b, reg):
    # print '--=------softmax_loss_vectorized----------------------------------------------'

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
    num_classes = W.shape[1]

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
    print 'y_hat'
    print y_hat

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

    return Z3, loss, grad, cache



def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

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






def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer

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

    for i in range(m):                       # loop over the training examples

        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i, :, :, :]

        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)

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
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev







def forward_propagation_vectorized(X, parameters, Y, reg):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

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

    Z1, cache_line_1 = conv_forward(X, W1, b1, hparameters)


    # relu
    A1, cache_activ_1 = relu(Z1)

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
    Z2, cache_line_2 = conv_forward(P1, W2, b2, hparameters)

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
    else:
        W3 = parameters['W3']
        b3 = parameters['b3']
    #         print '-------else----b3------------'
    #         print b3

        print '-----Z1.shape'
        print Z1.shape
        print '-----A1.shape'
        print A1.shape
        print '-----P1.shape'
        print P1.shape

        print '-----Z2.shape'
        print Z2.shape
        print '-----A2.shape'
        print A2.shape
        print '-----P2.shape'
        print P2.shape

        print '-----------W3------------'
        print W3.shape
        print '-----------P2_v------------'
        print P2_v.shape
        print '-----------b3------------'
        print b3.shape
    #     print b3

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



# GRADED FUNCTION: initialize_parameters

def initialize_parameters_pk(pkfile, load=False):
    """
    Initializes weight parameters to build a neural network with cnn. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
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


# GRADED FUNCTION: image2vector
def vector2image(vector, img_shape):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    (a, b, c, d) = img_shape
    ### START CODE HERE ### (≈ 1 line of code)
    img = vector.reshape((a, b, c, d))
    ### END CODE HERE ###

    return img


# GRADED FUNCTION: linear_forward

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

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
    y = np.where(Y == 1)[1]
    return y


# GRADED FUNCTION: random_mini_batches

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

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
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m / mini_batch_size)
    num_complete_minibatches = int(num_complete_minibatches)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:m, :]
        #         print '-----mini_batch_X-------'
        #         print mini_batch_X.shape
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def pad_backward(A_prev_pad, pad):
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



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    for i in range(1,4):
        parameters['W'+str(i)] = parameters['W'+str(i)] - learning_rate*grads['dW'+str(i)]
#         if(i==3):
#             grads['db'+str(i)].reshape(parameters['b'+str(i)].shape[0],parameters['b'+str(i)].shape[1] )
        parameters['b'+str(i)] = parameters['b'+str(i)] - learning_rate*grads['db'+str(i)]
#         print '--------grads[db+str(i)].shape----------'
#         print '--------parameters['b'+str(i)].shape----------'
#         print grads['db'+str(i)].shape
#         print parameters['b'+str(i)].shape
    return parameters


def cnn_model_backward(grad3, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

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

    dPad_2 = pool_backward(dPool_2, cache_pool_2, mode='max')

    dA2 = pad_backward(dPad_2, cache_pad_2)

    dZ2 = relu_backward(dA2, cache_activ_2)

    dPool_1, dW2, db2 = conv_backward(dZ2, cache_line_2)

    dPad_1 = pool_backward(dPool_1, cache_pool_1, mode='max')

    dA1 = pad_backward(dPad_1, cache_pad_1)

    dZ1 = relu_backward(dA1, cache_activ_1)

    dA0, dW1, db1 = conv_backward(dZ1, cache_line_1)

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
        'dW3': dW3,
        'db3': db3,
    }

    return grads


def cnn_batch_model(X, Y, parameters, reg=0.01, learning_rate=0.0075):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
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


def train_model_pk(pkfile,load, X, Y, reg=0.01, learning_rate=0.0075, num_epochs=100, minibatch_size=1, print_cost=True):
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

    # print parameters

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
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters



def cnn_model_train_test_pk(pkfile,load ,X_train, Y_train, X_test, Y_test, reg=0.01, learning_rate=0.01, num_epochs=40,
                            minibatch_size=2, print_cost=True):


    parameters = train_model_pk(pkfile,load, X_train, Y_train, reg, learning_rate, num_epochs, minibatch_size, print_cost)


    print '-------train------accuracy-----'
    accuracy = test_model(X_train, Y_train, parameters, minibatch_size)

    print '-------test------accuracy-----'
    accuracy = test_model(X_test, Y_test, parameters, minibatch_size)


def Y_hat2y_hat_ver(Y_hat):
    Y_hat = Y_hat.T

    print '----Y_hat------'
    print Y_hat
    Y_hat = np.where(Y_hat == np.max(Y_hat, axis=0))[0]

    Y_hat = np.array(Y_hat)
    # Y_hat = Y_hat.T
    y_hat = np.squeeze(Y_hat)

    return y_hat



def test_model_onces(X_test, Y_test, parameters):
    '''
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



def test_model(X_test, Y_test, parameters, minibatch_size=1):
    '''
    X_test ,(number,n_h,n_w,n_c)
    Y_test, (classes,number)

    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
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
    num_minibatches = int(m / minibatch_size)
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


pkfile = '/home/yanggang/longlongaaago/pycharm_cnn_v1/out/cnn_v1.pk'





import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
import scipy
from PIL import Image
from scipy import ndimage
from cnn_utils import *
from dnn_utils_v2 import *



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
# index = 6
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))


np.random.seed(1)
parameters = initialize_parameters_pk(pkfile,load=False)
# X = np.random.randn(5,64,64,3)
# Y = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,1,0],[1,0,0,0,0,0],[0,0,0,0,0,1]])

# loss,grad3,caches = forward_propagation_vectorized(X, parameters,Y,reg=0.01)


X_train_sub = X_train[0:10,:,:,:]
X_test_sub = X_test[0:10,:,:,:]
Y_train_sub = Y_train[0:10,:]
Y_test_sub = Y_test[0:10,:]
# accuracy = test_model(X, Y, parameters,minibatch_size=2)


# print loss

load = False
cnn_model_train_test_pk(pkfile,load ,X_train_sub, Y_train_sub, X_test_sub,Y_test_sub, reg=0.01, learning_rate=0.01, num_epochs=2,
                            minibatch_size=2, print_cost=True)