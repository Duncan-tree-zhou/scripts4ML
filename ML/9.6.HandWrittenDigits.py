#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn import svm
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def show_accuracy(a, b, tip):
    acc = a.ravel() ==b.ravel()
    print tip + '正确率：%.2f%%' % (100*np.mean(acc))

def save_image(im, i):
    im *= 15.9375
    im = 255 - im
    a = im.astype(np.uint8)
    output_path = '../save'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    Image.fromarray(a).save(output_path+'\\%d.png' %i)

if __name__ == "__main__":
    print 'Load Training File Start...'
    data = np.loadtxt('../datasets/16.optdigits.tra', dtype=np.float, delimiter=',')
    x,y = np.split(data,(-1,),axis=1)
    images = x.reshape(-1,8,8)
    y = y.ravel().astype(np.int)

    print 'Load Test Data Start...'
    data = np.loadtxt('../datasets/16.optdigits.tes', dtype=np.float, delimiter=',')
    x_test, y_test = np.split(data,(-1,), axis=1)
    images_test = x_test.reshape(-1,8,8)
    y_test = y_test.ravel().astype(np.int)
    print 'Load Data OK...'

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15,9), facecolor='w')
    for index, image in enumerate(images[:16]):
        plt.subplot(4, 8, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'训练图片： %i' % y[index])
    for index, image in enumerate(images_test[:16]):
        plt.subplot(4, 8, index + 17)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        save_image(image.copy(), index)
        plt.title(u'测试图片：%i' % y_test[index])
    plt.tight_layout()
    plt.show()

    # params = {'C':np.logspace(0, 3, 7), 'gamma':np.logsapce(-5, 0, 11)}
    # model = GridSearchCV(svm.SCV(kernel='rbf'),param_grid=params, cv=3)
    model = svm.SVC(C=10, kernel='rbf', gamma=0.001)
    print 'Start Learning...'
    t0 = time()
    model.fit(x,y)
    t1 = time()
    t = t1 - t0
    print '训练耗时：%d分钟%.3f秒' % (int(t/60), t - 60*int(t/60))
    # print '最优参数：\t', model.best_params_
    print 'Learning is OK...'
    y_hat = model.predict(x_test)
    show_accuracy(y,y_hat,'训练集')
    # print accuracy_score(y.ravel(), y_hat)
    y_hat = model.predict(x_test)
    print y_hat
    print y_test
    show_accuracy(y_test, y_hat, '测试集')

    err_images = images_test[y_test!= y_hat]
    err_y_hat = y_hat[y_test!=y_hat]
    err_y = y_test[y_test!=y_hat]
    print err_y_hat
    print err_y
    plt.figure(figsize=(10,8), facecolor='w')
    for index, image in enumerate(err_images):
        if index >= 12:
            break
        plt.subplot(3, 4, index +1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    plt.show()











