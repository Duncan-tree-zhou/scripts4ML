# -*- coding:utf-8 -*-
# /usr/bin/python

import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
from scipy import stats

if __name__ == "__main__":
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.sans-serif'] = 'SimHei'

    p1 = np.mat(np.array([30,20,1]))

    x1 = np.zeros(11)
    y1 = np.zeros(11)
    for i in range(0,11):
        T1 = np.mat(np.array([[1,0,i],[0,1,i],[0,0,1]]))
        tp = np.array(T1*np.transpose(p1))
        x1[i]=tp[0,0]
        y1[i]=tp[1,0]
    print x1
    print y1
    plt.plot( x1, y1,'bo-',label='T1*x')


    x2 = np.zeros(10)
    y2 = np.zeros(10)
    theta = 60*math.pi/180
    N=10
    for i in range(0,N):
        a1 = math.cos(theta*i/N)
        a2 = -math.sin(theta*i/N)
        b1 = math.sin(theta*i/N)
        b2 = math.cos(theta*i/N)
        T2 = np.mat(np.array([[a1,a2,0],[b1,b2,0],[0,0,1]]))
        tp2 = np.array(T2*np.transpose(p1))
        x2[i]=tp2[0,0]
        y2[i]=tp2[1,0]
    print x2
    print y2
    plt.plot( x2, y2,'go-',label='T2*x')

    plt.plot( p1[0,0], p1[0,1],'ro',label='x')
    plt.legend(loc='lower right')
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.title(u'矩阵乘法几何意义', fontsize=16)
    plt.grid(b=True)
    plt.show()