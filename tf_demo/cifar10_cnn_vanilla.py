import tf_demo.models.tutorials.image.cifar10 as cifar10,tf_demo.models.tutorials.image.cifar10_input as cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev = stddev))
    if w1 is not None:
        weight_loss = tf.multifply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

cifar10.maybe_download_and_extract()

images_train, labels_train = cifar10_input.distorted_input(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 4])
label_holder = tf.placeholder(tf.int32, [batch_size])

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev = 5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0 , beta = 0.75)

weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1,1,1,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))

conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim,384],stddev = 0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)





