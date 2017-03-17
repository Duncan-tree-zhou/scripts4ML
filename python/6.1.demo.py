#!/usr/bin/python
# -*- coding:utf-8 -*-

# 导入NumPy函数库，一般都是用这样的形式(包括别名np，几乎是约定俗成的)

import numpy as np
import matplotlib as mpl


if __name__ == "__main__":
    # # 开场白：
    # numpy是非常好用的数据包，如：可以这样得到这个二维数组
    # [[ 0  1  2  3  4  5]
    #  [10 11 12 13 14 15]
    #  [20 21 22 23 24 25]
    #  [30 31 32 33 34 35]
    #  [40 41 42 43 44 45]
    #  [50 51 52 53 54 55]]
    #a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
    #print a
    b = np.arange(0, 70, 10).reshape((-1,1))
    c = np.arange(6)
    print b
    print c
    print __name__