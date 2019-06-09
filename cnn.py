from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
#权重函数
def weight_variable(shape):
    # 产生随机变量
    # truncated_normal：选取位于正态分布均值=0.1附近的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

 #偏置值函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
#输入x是图片信息矩阵，W是卷积核的值
def conv2d(x, W):
    # stride = [1,水平移动步长,竖直移动步长,1]
    # 第二个参数表示：卷积核每次向右移动的步长 第三个参数表示：卷积核每次向下移动的步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
    # stride = [1,水平移动步长,竖直移动步长,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
#MNIST是Google的很经典的一个做图像识别的数据集，图片大小是28*28的
#读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
sess = tf.InteractiveSession()
 
#预定义输入值X、输出真实值Y    placeholder为占位符
#shape=[None, 784]是数据维度大小，将28*28的二维数据转换成一个一维的、长度为784的新向量
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32) #keep_prob 是改变参与计算的神经元个数的值
x_image = tf.reshape(x, [-1,28,28,1]) #图片集是黑白单色，x_image 中的图片尺寸参数最后一个 = 1，彩色 = 3

 
#卷积层1网络结构定义
#卷积核1：patch=5×5;in size 1;out size 32;激活函数reLU非线性处理
W_conv1 = weight_variable([5, 5, 1, 32]) 
#这里的卷积核大小是5*5的，输入的通道数是1，输出的通道数是32
#卷积核的值这里就相当于权重值，用随机数列生成的方式得到   
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                          #output size 14*14*32
#由于MNIST数据集图片大小都是28*28，且是黑白单色
#所以准确的图片尺寸大小是28*28*1（1表示图片只有一个色层，彩色图片都是3个色层——RGB）
#所以经过第一次卷积后，输出的通道数由1变成32，图片尺寸变为：28*28*32（相当于拉伸了高）
#再经过第一次池化（池化步长是2），图片尺寸为14*14*32

#卷积层2网络结构定义 
#卷积核2：patch=5×5;in size 32;out size 64;激活函数reLU非线性处理
#卷积核大小是5*5的，输入的通道数是32，输出的通道数是64
W_conv2 = weight_variable([5, 5, 32, 64])     
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)   #output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                            #output size 7 *7 *64
 


# 全连接层1
W_fc1 = weight_variable([7*7*64,1024]) 
#全连接层的输入就是第二次池化后的输出，尺寸是7*7*64，全连接层1有1024个神经元
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) #[n_samples,7,7,64]->>[n_samples,7*7*64] 
#tf.reshape(a,newshape)函数，当newshape = -1时，函数会根据已有的维度计算出数组的另外shape属性值
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 
# 减少计算量dropout.  keep_prob 是为了减小过拟合现象。
# 每次只让部分神经元参与工作使权重得到调整。
 
# 全连接层2
W_fc2 = weight_variable([1024, 10])      
#全连接层2有10个神经元，相当于生成的分类器
b_fc2 = bias_variable([10])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2   
#经过全连接层1、2，得到的预测值存入prediction 中
 

#二次代价函数:预测值与真实值的误差
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
 
#梯度下降法:数据太庞大,选用AdamOptimizer优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
 
#结果存放在一个布尔型列表中
#tf.argmax(prediction,1)返回的是对于任一输入x预测到的标签值，tf.argmax(y_,1)代表正确的标签值
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#correct_prediction 这里是返回一个布尔数组。
#为了计算我们分类的准确率，我们将布尔值转换为浮点数来代表对与错，然后取平均值。
#例如：[True, False, True, True]变为[1,0,1,1]，计算出准确率就为0.75 

saver = tf.train.Saver()  # defaults to saving all variables
sess.run(tf.global_variables_initializer())
 

for i in range(1000):
    batch = mnist.train.next_batch(50)  #batch 是来源于MNIST数据集，一个批次包含50条数据
    if i % 100 == 0: 
#feed_dict=({x: batch[0], y_: batch[1], keep_prob: 0.5}语句：是将batch[0]，batch[1]代表的值传入x，y_；
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step", i, "training accuracy", train_accuracy)

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
 
#保存模型参数
saver.save(sess, './model.ckpt')


