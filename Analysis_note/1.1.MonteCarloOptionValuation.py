# -*- coding: utf-8 -*-

from __future__ import division, print_function
from math import log, sqrt, exp
from scipy import stats

from time import time
from random import gauss, seed
import numpy as np

def bsm_call_values(S0, K, T, r, sigma):
    """
        根据BSM公式计算期权估值

        参数
        ======
        S0:     初始标的物价格，即t=0
        K:      期权行权价格
        T:      期权到期日
        r:      固定无风险短期利率
        sigma:  标的物固定波动率

        返回值
        ======
        value:  当前期权定价
    """
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0)
    return value

def run_main():
    S0 = 100.
    K = 105.
    T = 1.
    r = 0.05
    sigma = 0.2
    init_value = bsm_call_values(S0, K, T, r, sigma)
    print('BSM方法的期权估值：', init_value)

    M = 50  # 子时段个数
    dt = T / M  # 子时段时间间隔
    I = 250000  # 迭代次数

    # 方法1. 纯Python，只
    t0 = time()
    S = []
    for i in range(I):
        path = []
        for t in range(M + 1):
            if t == 0:
                path.append(S0)
            else:
                z = gauss(0., 1.)
                S_t = path[t - 1] * exp((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z)
                path.append(S_t)
        S.append(path)
    C_0 = exp(-r * T) * sum([max(path[-1] - K , 0) for path in S]) / I

    duration = time() - t0
    print('使用纯Python实现期权估值的模拟：', C_0)
    print('耗时{}秒'.format(duration))


    # 方法2. 向量化NumPy, 使用NumPy功能实现更加紧凑搞笑的本本
    t1 = time()
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        z = np.random.standard_normal(I)
        S[t] = S[t - 1] * np.exp((r - 0,5 * sigma ** 2) * dt + sigma * sqrt(dt) * z)
    C_0 = exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I

    duration2 = time() - t1
    print('使用NumPy实现期权估值模拟：', C_0)
    print('耗时{}秒'.format(duration2))



if __name__ == "__main__":
    run_main()