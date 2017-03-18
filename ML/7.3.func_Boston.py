#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    path = '../datasets/7.housing.data'
    f = file(path)
    # 选取列
    cols = np.linspace(0,12,13,dtype=int)
    # print cols
    # cols = [1,4,5,7,8,9,10,12]
    # cols = [0,13]
    x = []
    y = []
    for i,d in enumerate(f):
        d = d.strip()
        if d:
            tmp = d.split()
            x.append([tmp[j] for j in cols])
            y.append(tmp[-1])
    x = np.array(x,dtype='float64')
    y = np.array(y,dtype='float64')

    print u'样本个数：%d, 特征个数：%d' % x.shape

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=1)

    model = Pipeline([('sc',StandardScaler()),
                      ('poly',PolynomialFeatures()),
                      ('linear',ElasticNetCV(alphas=np.logspace(-3,5,10),l1_ratio=np.linspace(0,1,5),cv=3,fit_intercept=False))])

    print u'开始建模...'
    model.set_params(poly__degree=1)
    model.fit(x_train,y_train.ravel())
    linear = model.get_params('linear')['linear']
    print u'参数：', linear.coef_
    print u'超参数：', linear.alpha_
    print u'L1 ratio：', linear.l1_ratio_
    # print u'系数：', linear.coef_.ravel()
    y_pred = model.predict(x_test)
    print x_test.shape
    print y_test.shape
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print 'R2:', r2
    print u'均方误差：', mse

    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test.ravel(), 'r-', lw=2, label=u'真实值')
    plt.plot(t, y_pred, 'g-', lw=2, label=u'估计值')
    plt.legend(loc='best')
    plt.title(u'波士顿房价预测', fontsize=18)
    plt.xlabel(u'样本编号', fontsize=15)
    plt.ylabel(u'房屋价格', fontsize=15)
    plt.grid()
    plt.show()

    plt.figure(figsize=(15,12))
    for i in range(len(cols)):
        plt.subplot(7,2,(i+1))
        print x.shape
        x_tmp = x[:,i]
        plt.plot(x_tmp,y,'ro')
        plt.title(u'第%d列'%(i+1))
        plt.grid()
    plt.show()




