#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = '../datasets/7.iris.data'
    df = pd.read_csv(path,header=None)
    x = df.values[:,:-1]
    y = df.values[:,-1]
    y = LabelEncoder().fit_transform(y)
    print df
    print 'x=\n',x
    print 'y=\n',y

