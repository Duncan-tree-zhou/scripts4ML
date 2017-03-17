#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    x = np.arange(0.05,3,0.05)

    y3 = [math.log(a,3) for a in x]
    plt.plot( x, y3, linewidth=2, color="#F75000", label="log3(x)")

    y2 = [math.log(a,2) for a in x]
    plt.plot( x, y2, linewidth=2, color="#9F35FF", label="log2(x)")

    y1 = [math.log(a,1.5) for a in x]
    plt.plot( x, y1, linewidth=2, color="#007500", label="log1.5(x)")
    a = x[0.25<=x]
    a = a[a<=1.75]
    b = math.log(math.e,1.5)*(a-1)
    plt.plot( a, b, "--", linewidth=2, color="#007500", label="tangent of log1.5(x) at (1,0)")


    plt.plot([1, 1], [y1[0], y1[-1]], "r:", linewidth=2)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()