# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

if __name__ == "__main__":
    stock_max, stock_min, stock_close, stock_amount \
        = np.loadtxt('7.SH600000.txt', delimiter='\t', skiprows=2, usecols=(2, 3, 4, 5), unpack=True)
    N = 100
    stock_close = stock_close[:N]
    print stock_close

    n = 5
    weight = np.ones(n)
    weight /= weight.sum()
    print weight
    stock_sma = np.convolve(stock_close,weight,mode='valid')