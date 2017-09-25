#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        # 输入维度数
        self.n_input = n_input
        # 隐藏层数
        self.n_hidden = n_hidden
        # 激活函数，这里用softplus
        self.transfer = transfer_function
        # 高斯噪声系数
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        # 参数w1,b1.w2.b2
        network_weights = self._initialize_weights()
        self.weights = network_weights
        # 输入x
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 从输入到隐藏层运算
        self.hidden = self.transfer(tf.add(
            tf.matmul(
                self.x + scale * tf.random_normal((n_input,)),
                self.weights['w1']),
            self.weights['b1']))
        # 重建，即输出
        self.reconstruction = tf.add(
            tf.matual(
                self.hidden,
                self.weights['w2']),
            self.weights['b2'])
        # 损失函数（最小二乘）
        self.cost = 0.5 * tf.reduce_sum(
            tf.pow(
                tf.subtract(
                    self.reconstruction,
                    self.x
                ),
                2.0))
        # 优化器
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input],dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.zeros([self.n_input],dtype = tf.float32)]))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run(
            (self.cost, self.optimizer),
            feed_dict = {self.x:X, self.scale: self.training_scale }
        )
        return cost

    def calc_total_cost(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict= {self.x: X, self.scale:self.training_scale})
        return cost

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict= {self.x: X, self.scale: self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size= self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict= {self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x:X, self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])



if __name__ == '__main__':
