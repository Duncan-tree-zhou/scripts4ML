#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pydotplus
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 人工制造数据
    N = 100
    x = np.random.rand(N) * 6 - 3 # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    print y
    x = x.reshape(-1,1) # 转置后，得到N个样本，每个样本都是1维的
    print x

    model = DecisionTreeRegressor(criterion='mse', max_depth=9)
    model.fit(x,y)
    x_test = np.linspace(-3,3,50).reshape(-1,1)
    y_hat = model.predict(x_test)
    plt.plot(x, y, 'r*', ms=10, label='Actual')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # 比较决策树深度影响

    depths = [3,5,7,9,11]
    clr = 'rgbmy'
    plt.plot(x, y, 'k^', linewidth=2, label='Actual')
    x_test = np.linspace(-3,3,50).reshape(-1,1)
    for i, depth in enumerate(depths):
        model = DecisionTreeRegressor(criterion='mse', max_depth=depth)
        model.fit(x,y)
        y_hat = model.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=clr[i], linewidth=2, label='Depth=%d' % depth)
    plt.legend(loc = 'upper left')
    plt.grid()
    plt.show()