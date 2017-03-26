#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
import numpy as np
import matplotlib.pyplot as plt


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = '../datasets/7.iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x_prime, y = np.split(data, (4,), axis=1)

    iris_feature = [ u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度']
    feature_pain = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    # 分开测试与训练数据
    x_train, x_test, y_train, y_test = train_test_split(x_prime,y,train_size=0.7,random_state=1)

    plt.figure(figsize=(12, 10), facecolor='#FFFFFF')
    for i,pair in enumerate(feature_pain):
        # 准备数据
        x_train_tmp = x_train[:,pair]
        x_test_tmp = x_test[:,pair]

        # 决策树学习
        model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
        dt_clf = model.fit(x_train_tmp,y_train)
        y_train_hat_tmp = model.predict(x_train_tmp)
        y_test_hat_tmp = model.predict(x_test_tmp)
        y_train_hat_tmp = y_train_hat_tmp.reshape((-1,1))
        y_test_hat_tmp = y_test_hat_tmp.reshape((-1,1))
        c_train = float(np.count_nonzero(y_train_hat_tmp == y_train))
        c_test = float(np.count_nonzero(y_test_hat_tmp == y_test))
        print '特征：', iris_feature[pair[0]] + ' + ' + iris_feature[pair[1]]
        print '训练集准确率：%.2f/%.2f = %.2f%%'%(c_train,len(y_train_hat_tmp),(100*c_train/len(y_train_hat_tmp)))
        print '测试集准确率：%.2f/%.2f = %.2f%%'%(c_test,len(y_test_hat_tmp),100*c_test/len(y_test_hat_tmp))



        # 画图
        N, M = 500, 500 # 横纵各采样多少个值
        x1_min, x1_max = min(x_train_tmp[:,0].min(),x_test_tmp[:,0].min()), max(x_train_tmp[:,0].max(),x_test_tmp[:,0].min()) # 第0列的范围
        x2_min, x2_max = min(x_train_tmp[:,1].min(),x_test_tmp[:,1].min()), max(x_train_tmp[:,1].max(),x_test_tmp[:,1].min()) # 第1列的范围
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2) # 生成网络采样点
        x_grid = np.stack((x1.flat, x2.flat), axis=1)
        y_grid = model.predict(x_grid).reshape((N,M))
        # 显示
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark_train = mpl.colors.ListedColormap(['g', 'r', 'b'])
        cm_dark_test = mpl.colors.ListedColormap(['g', 'r', 'b'])
        plt.subplot(2, 3, i+1)
        plt.pcolormesh(x1, x2, y_grid, cmap=cm_light)  # 结果模型分类区域视图
        plt.scatter(x_train_tmp[:, 0], x_train_tmp[:, 1], marker='o', c=y_train_hat_tmp, edgecolors='k', cmap=cm_dark_train, label=u'训练')  # 训练
        plt.scatter(x_test_tmp[:, 0], x_test_tmp[:, 1], marker='^', c=y_test_hat_tmp, edgecolors='k', cmap=cm_dark_test, label=u'测试')  # 测试
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
        # plt.legend('lower right')
        # if i==3:
        #     plt.legend('lower left')
    plt.suptitle(u'决策树对鸢尾花数据的两特征组合的分类结果', fontsize=18)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.92)
    plt.show()

