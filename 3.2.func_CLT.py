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

    #均匀分布求均值服从正态分布
    u1 = np.random.uniform(0.0, 1.0, 10000)
    plt.figure(figsize=(15,8), facecolor="w")
    plt.subplot(221)
    plt.hist(u1,50,facecolor="g",alpha=0.75)
    plt.title(u'均匀分布试验', fontsize=16)
    plt.grid(True)

    times = 10000
    # 横向求取均值
    u2 = np.random.uniform(0.0, 1.0, 10000)
    for time in range(times):
        u2 += np.random.uniform(0.0, 1.0, 10000)
    print len(u2)
    u2 /= times

    # 纵向求取均值
    # u2 = []
    # for time in range(times):
    #     tmp = np.random.uniform(0.0, 1.0, 10000)
    #     u2.append(tmp.sum()/len(tmp))


    plt.subplot(222)
    plt.hist(u2, 80, facecolor='g', alpha=0.75)
    plt.title(u'均匀分布的均值分布', fontsize=16)
    plt.grid(True)


    #泊松分布均值服从正态分布
    lamda = 10
    p = stats.poisson(lamda)
    y = p.rvs(size=1000)
    print y
    mx = 30
    r = (0, mx)
    print r
    bins = r[1] - r[0]
    print bins

    plt.subplot(223)
    plt.hist(y,bins=bins,range=r,color='g',alpha=0.8,normed=True)
    t= np.arange(0,mx)
    plt.plot(t,p.pmf(t),'ro-', lw=2)
    plt.title(u'泊松分布', fontsize=16)
    plt.grid(True)

    N = 1000
    M = 10000
    plt.subplot(224)
    a = np.zeros(M, dtype=np.float)
    p = stats.poisson(lamda)
    for i in np.arange(N):
        y = p.rvs(size=M)
        a += y
    a /= N
    plt.hist(a, bins=20, color='g', alpha=0.8)
    plt.title(u'泊松分布的均值分布', fontsize=16)
    plt.grid(b=True)
    plt.show()

