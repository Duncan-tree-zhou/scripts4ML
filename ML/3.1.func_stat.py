# -*- coding:utf-8 -*-
# /usr/bin/python

import matplotlib.pyplot as plt
import math
import numpy as np

def calc(data):
    n = len(data)
    niu = 0.0
    niu2 = 0.0
    niu3 = 0.0
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu /= n
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu*niu)
    return [niu,sigma,niu3]

def calc_stat(data):
    [niu, sigma, niu3] = calc(data)
    n = len(data)
    niu4 = 0.0
    for a in data:
        a -= niu
        niu4 += a ** 4
    niu4 /= n

    skew = (niu3 - 3*niu*sigma**2-niu**3)/(sigma**3)
    hurt = niu4 / (sigma**4)
    return [niu, sigma, skew, hurt]

if __name__ == "__main__":
    data = list(np.random.randn(10000))
    data2 = list(2*np.random.randn(10000))
    data3 = [x for x in data if x > -0.5]
    data4 = list(np.random.uniform(0,4,10000))

    [niu,sigma,skew,hurt] = calc_stat(data)
    [niu2,sigma2,skew2,hurt2] = calc_stat(data2)
    [niu3,sigma3,skew3,hurt3] = calc_stat(data3)
    [niu4,sigma4,skew4,hurt4] = calc_stat(data4)
    print niu,sigma,skew,hurt
    print niu2,sigma2,skew2,hurt2
    print niu3,sigma3,skew3,hurt3
    print niu4,sigma4,skew4,hurt4

    print data
    print "===="
    print data2

    info = r'$\mu=%.2f, \sigma=%.2f,\ skew=%.2f,\ hurt=%.2f$' % (niu,sigma,skew,hurt)
    info2 = r'$\mu=%.2f, \sigma=%.2f,\ skew=%.2f,\ hurt=%.2f$' % (niu4,sigma4,skew4,hurt4)

    plt.text(1,0.38,info,bbox=dict(facecolor='red',alpha=0.25))
    plt.text(1,0.35,info2,bbox=dict(facecolor='green',alpha=0.25))

    plt.hist(data, 50, normed=1, facecolor='r', alpha=0.9)
    plt.hist(data4, 80, normed=1, facecolor='g', alpha=0.9)

    plt.grid(True)
    plt.show()