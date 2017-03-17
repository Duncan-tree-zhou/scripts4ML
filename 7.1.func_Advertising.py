#!/usr/bin/python
#  -*- coding:utf-8 -*-

from pprint import pprint
import numpy as np
import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV,ElasticNetCV,ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures as plyF
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.set_printoptions(suppress=True)
    path = './datasets/7.Advertising.csv'
    # f = file(path)
    # x = []
    # y = []
    # for i,d in enumerate(f):
    #     if i==0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = map(float,d.split(','))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # x = np.array(x)
    # y = np.array(y)
    # pprint(x)
    # pprint(y)

    #python自带
    # f = file(path,'r')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()
    # print '=============='

    # numpy 读入
    # p = np.loadtxt(path,delimiter=',',skiprows =1)
    # print p

    #pandas读入
    data = pd.read_csv(path)
    # print data.head()
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data[['Sales']]
    # print x.shape
    # print y.shape
    # print x
    # print y

    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=[10,10])
    plt.subplot('311')
    plt.grid(True)
    plt.plot(data['TV'],y,'ro',label='TV')
    plt.legend(loc='lower right')
    plt.subplot('312')
    plt.grid(True)
    plt.plot(data['Radio'],y,'g^',label='Radio')
    plt.legend(loc='lower right')
    plt.subplot('313')
    plt.grid(True)
    plt.plot(data['Newspaper'],y,'mv',label='Newspaper')
    plt.legend(loc='lower right')
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state = 1)
    print x_train.shape, y_train.shape
    alpha = np.logspace(-3,5,10)

    # model1 = ElasticNetCV(alphas=alpha,l1_ratio=[.1,.5,.7,.9,.95,.99,1],cv=4)
    model1 = LassoCV(alphas=alpha,cv=4)
    model = Pipeline([('poly',plyF()),('linear',model1)])

    # 逐一升维求解
    N = 8
    plt.figure(figsize=[15, 10])
    for i in range(N):

        model.set_params(poly__degree=i+1)
        model.fit(x_train,y_train)
        lin = model.get_params('linear')['linear']
        # print 'alpha=%.6f,l1_ratio=%.6f,coef=%s'%(lin.alpha_,lin.l1_ratio_,lin.coef_)
        print 'alpha=%.6f,coef=%s'%(lin.alpha_,lin.coef_)

        y_hat = model.predict(x_test)
        y_hat.shape = -1,1
        # print y_hat.shape
        # print y_test.shape
        mse = np.average((y_hat-y_test)**2)
        rmse = np.sqrt(mse)
        print 'i = ',i+1
        print 'MSE = ',mse
        print 'RMSE = ',rmse
        print 'R2-train = ',model.score(x_train,y_train)
        print 'R2-test = ',model.score(x_test,y_test)
        print '====================='

        #真实数据与测试数据做对比
        plt.subplot('%d2%d'%((N+1)/2,i+1))
        t = np.arange(len(x_test))
        plt.plot(t,y_test,'r-',linewidth=2,label=u'测试数据')
        plt.plot(t,y_hat,'g-',linewidth=2,label=u'预测数据')
        # plt.title(u'%d阶线性回归预测销量'%(i+1),fontsize=5)
        plt.legend(loc='lower right')
        plt.grid()
    plt.show()



