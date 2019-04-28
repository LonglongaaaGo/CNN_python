# CNN_python

实现了卷积、全连接、下采样、relu、sigmod、softmax、等等前向、反向传播，将6类车标作为输入进行模型训练，不过是CPU版本的

### 基于车标的 6分类模型  
两层卷积+relu +下采样+ 1层全连接+softmax_with_loss  
具体参数看代码  
编写环境  python 2.7   pycharm 2017（没什么影响）  

一、脚本：  
1.Analyze_.py  进行每次训练之后的  的cost,准确率等分析脚本 
 
2.cnn_fun.py        主要的卷积神经网络的方法脚本，有很多冗余，基本上有好几个版本在里面，相同后缀名的标示相同版本  
正常命名的表示最初版本_单线程，有后缀的表示迭代版本，说明如下：  
如   
（1）conv_forward   标示正常版本，及卷积的前向传播  
（2）conv_forward_test  表示测时版本，但是不一定对，可能有其他功能，注意 看注释  
（3）conv_forward_mul   表示多进程   ，但是废弃不用，因为进程池没有管理者，不能保证程序正常运行  
（4）conv_forward_mul_ext 表示多进程 ，可以使用，因为进程池有管理者，能正常运行  
（5）_pk   后缀名的方法表示，有保存或读取pikcle 文件的操作  
（6）_vectorized  后缀名的方法 ，只是仅仅维度可能有转置操作，导致维度变化  
×××主要的调用和使用还是从主程序出发×××  
目前使用的是含有 （4）、（5）、（6）后缀的方法， 是基于多进程的，所以内存占用会很大，要注意减小batchsize （main函数）  
3.dnn_utils_v2.py  一些基本的激活方法   
4.h5file.py   用于将数据生成对应的  h5 文件，方便压缩读取    和程序运行，具体操作看脚本注释  
5.main.py  程序的主要入口，可以直接跑，从main方法里面逐步跟踪，了解调用了哪些函数  
6.test_batches.py  主要进行最后模型训练出来之后，用测试集进行分析的脚本，具体操作看注释  
7.test_forward.py  主要实现了单线程的测试工作   不过这个只能参考   在 cnn_window   项目中有细微改动，以那个为标准  
同时要注意，在拿来展示的时候，就是放在其他系统上或者展示，一定要注意用单线程的  
×××未列出的脚本都不重要，没有被调用，可以被忽略，但是有一些可以参考  

二、数据  
Cardataset、CardatasetV2  文件夹  
data_ori 表示存放了原始的图片文件  
datah5  表示存放了将原始文件生成的h5文件  

三、标签  
label  文件夹  
label.txt  直接对应分类的标签  ，加后缀表示版本不同  
train_map.txt   文件全路径 和标签映射 加后缀表示版本不同   在训练过程中用于监督学习  
test_map.txt    文件全路径 和标签映射 加后缀表示版本不同  在测试过程中用于计算准确率  

四、模型输出  
out 文件夹  
cnnv1car_0.pk  后面数字逐渐递增，表示，从当前模型开始，数字在这个参数的基础上，逐渐+1，并切保存下来
保存的区间时 1次 epoch 保存1次  
不同版本文件夹会不同  
五、过程记录  
processing  文件夹  
手动记录过程，以及损失值、准确率的走势  

六、最后测试集的模型结果分析  
result 文件夹  有txt文件和pk 文件，是对应test_batches.py所生成的数据    详情看对应脚本注释  

七、其他信息  
other_para  文件夹  
用于保存在训练过程中的 损失值、准确率等在训练模型的过程中自动生成  
然后，用Analyze_.py 将这些文件进行分析  对其进行图标的绘制  

八、系统文件  
venv  文件夹    
项目给予  pycharm  来进行，其实这个文件不是特别重要  
可以直接在python 2.7 下跑  


×××理解本项目方法×××  
从main 脚本开始，进行每一步程序的调用跟踪，来了解细节，这里实现了多进程即pool = ProcessPoolExecutor() ,如果向换成单线程的  
可以进行从后缀名下手更改   

### 运行方法  
注意：如果报错了，一般都是缺某个文件夹或者文件，不要担心，可能是你没有将数据生成hdf5格式  
1.首先你要生成lable文件，运行h5file.py  参考注释  
2.如果没有hdf5文件 则先生成 运行h5file.py，可以参考注释文件  （1,2两步同步进行）  
3.在main.py设置对应的超参数  
4.run main.py  
5.在对应的结果文件夹内，会有模型和参数等保存  
6.调用Analyze_.py 进行分析   
### 如果有什么问题请及时和我联系！！欢迎交流！！







