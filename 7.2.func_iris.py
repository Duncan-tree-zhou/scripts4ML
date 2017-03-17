#!/usr/bin/python
#  -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from matplotlib import pyplot as plt

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    path = './datasets/7.iris.data'
    # file读取
    # f = file(path)
    # x = []
    # y = []
    # for d in f:
    #     d = d.strip()
    #     if d:
    #         d = d.split(',')
    #         x.append(map(float,d[:-1]))
    #         y.append(d[-1])
    # x = np.array(x)
    # y = np.array(y)
    # print x
    # print y
    # y[y=='Iris-setosa']=0
    # y[y=='Iris-versicolor']=1
    # y[y=='Iris-virginica']=2
    # y = y.astype(dtype=np.int)
    # print y

    # pandas读取
    data = pd.read_csv(path,header=None)
    print data
    #直接转为np数组
    x = data.values[:,:-1]
    y = data.values[:,-1]
    print x.shape
    print y.shape
    print 'x=',x
    print 'y=',y
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print le.classes_
    print 'last version, y =\n',y

    x = x.astype(np.float)
    y = y.astype(np.int)

    # 仅使用前两列特征
    x = x[:,:2]
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=1)
    model = Pipeline([('sc',StandardScaler()),('clf',LogisticRegression())])
    model.fit(x_train,y_train.ravel())
    y_test_hat = model.predict(x_test)
    y_train_hat = model.predict(x_train)

    print u'测试集准确率：%.2f'%(np.average(y_test_hat==y_test.ravel()))
    print u'训练集准确率：%.2f'%(np.average(y_train_hat==y_train.ravel()))

    # 画图
    N, M = 500, 500 # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max() # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max() # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1,t2)                   # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1) # 测试点

    # print x1_min,x1_max,x2_min,x2_max
    # print 'x1=\n', x1
    # print '====='
    # print 'x2=\n', x2
    # print '====='
    # print 'x_test=\n', x_test
    # print '====='

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g','r','b'])
    y_hat = model.predict(x_test)
    y_hat = y_hat.reshape(x1.shape)


