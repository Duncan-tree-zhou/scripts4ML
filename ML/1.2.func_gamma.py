# -*- coding:utf-8 -*-
# /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gamma
from scipy.special import factorial


mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = 'SimHei'

if __name__ == "__main__":

    N = 2
    x1 = np.linspace(0,N,50)
    y1 = gamma(x1+1)
    plt.plot(x1, y1, 'r-',x1, y1, 'm*', lw=1)
    x2 = np.arange(0,N,1)
    y2 = factorial(x2)
    plt.plot(x2, y2, 'bo', lw=2)
    plt.xlim(-0.1,N+0.1)
    plt.ylim(0.5, np.max(y1)*1.05)
    plt.grid(b=True)
    plt.xlabel(u'X', fontsize=15)
    plt.ylabel(u'Gamma(X) - 阶乘', fontsize=15)
    plt.title(u'阶乘和Gamma函数', fontsize=16)
    plt.show()