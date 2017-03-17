# -*- coding:utf-8 -*-
# /usr/bin/python

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import perm

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = 'SimHei'

def count_birthday_duplicate_rate(x):
    return count_duplicate_rate(x,365)

def count_duplicate_rate(x,total):
    return 1-(perm(total,x)/np.power(total,x))

if __name__ == "__main__":

    N = 80
    # x = np.linspace(0,N,(N/10)+1)
    # y = count_birthday_duplicate_rate(x)
    # print x
    # print y
    x = np.linspace(0,N,N+1)
    y = count_birthday_duplicate_rate(x)
    plt.plot(x, y, 'b-', label="P(n)")

    y = y[y>0.5]
    x = x[-np.alen(y):]
    plt.plot(x, y, 'r*', linewidth=2, color="#F75000", label="P>50%")

    y = y[y>0.99]
    x = x[-np.alen(y):]
    plt.plot(x, y, 'ro', linewidth=2, label="P>99%")

    plt.grid(b=True)
    plt.xlabel(u'班级人数', fontsize=15)
    plt.ylabel(u'概率', fontsize=15)
    plt.title(u'至少两个人同一天生日的概率情况', fontsize=16)
    plt.legend(loc='lower right')
    plt.show()

