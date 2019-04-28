from cnn_fun import *

#直接进行测试的文件
def final_test():
    pkfile = '/home/yanggang/longlongaaago/pycharm_cnn_v1/out/cnn_v2_20180426.pk'
    load = True

    #     fname = "images/2018-04-28 14-14-32屏幕截图.png"
    #     fname = "images/2018-04-28 14-14-23屏幕截图.png"
    #     fname = "images/2018-04-28 14-14-13屏幕截图.png"
    #     fname = "images/2018-04-28 14-13-55屏幕截图.png"
    #     fname = "images/thumbs_up.jpg"
    #     fname = "images/20171113222231261.png"
    fname = "images/20171016212808289.png"
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64))
    plt.imshow(my_image)

    print my_image.shape

    if my_image.shape[2] > 3:
        new_my_image = my_image[:, :, 0:3]
    else:
        new_my_image = my_image
    print new_my_image.shape

    test = new_my_image.reshape(1, 64, 64, 3)
    print test.shape

    parameters = initialize_parameters_pk(pkfile, load)
    X_train = test / 255.
    y_hat = test_model_onces_forward(X_train, parameters)
    return y_hat


final_test()
