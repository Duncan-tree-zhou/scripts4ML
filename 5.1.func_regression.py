#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import warnings

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

def xss(y,y_hat):
    y = y.ravel()
    y_hat = y_hat.ravel()
    tss = ((y - np.average(y))**2).sum()
    rss = ((y_hat - y)**2).sum()
    ess = ((y_hat - np.average(y))**2).sum()
    r2 = 1 - rss / tss

    # print 'RSS:', rss, '\t ESS:', ess
    # print 'TSS:', tss, 'RSS + ESS = ', rss + ess

    tss_list.append(tss)
    rss_list.append(rss)
    ess_list.append(ess)
    ess_rss_list.append(rss + ess)
    corr_coef = np.corrcoef(y, y_hat)[0, 1]

    return r2, corr_coef



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.set_printoptions(suppress=True)
    np.random.seed(10)
    N = 9
    x = np.linspace(0,6,N) + np.random.randn(N)
    x = np.sort(x)
    np.random.seed(0)
    y = 2*x**2 - 5*x + 3 + np.random.randn(N)
    x.shape = -1,1
    y.shape = -1,1
    print x.shape
    print y.shape
    models = [
        Pipeline([('poly',PolynomialFeatures()),('linear',LinearRegression(fit_intercept=False))]),
        Pipeline([('poly',PolynomialFeatures()),('linear',RidgeCV(alphas=np.logspace(-3,2,50),fit_intercept=False))]),
        Pipeline([('poly',PolynomialFeatures()),('linear',LassoCV(alphas=np.logspace(-3,2,50),fit_intercept=False))]),
        Pipeline([('poly',PolynomialFeatures()),('linear',ElasticNetCV(alphas=np.logspace(-3,2,50),l1_ratio=[.1,.5,.7,.9,.95,.99,1]))])
    ]
    np.set_printoptions(suppress=True)
    clrs = []  # 颜色
    for c in np.linspace(16711680, 255, N-1):
        clrs.append('#%06x' % c)
    line_width = np.linspace(5,2,N)
    titles = [u'线性回归',u'Ridge回归',u'Lasso',u'ElasticNet']
    plt.figure(figsize=(18, 12), facecolor='w')
    tss_list = []
    rss_list = []
    ess_list = []
    ess_rss_list = []
    for i,model in enumerate(models):
        plt.subplot(2,2,i+1)
        plt.plot(x,y,'ro',ms=10,zorder=N)
        for j in range(N-1):
            model.set_params(poly__degree=(j+1))
            model.fit(x,y.ravel())
            x_hat = np.linspace(x.min(), x.max(),100)
            x_hat.shape = -1,1
            y_hat = model.predict(x_hat)
            s = model.score(x,y)
            lin = model.get_params('linear')['linear']

            output = u'%s：%d阶，系数为：' % (titles[i], j+1)
            if hasattr(lin, 'alpha_'):
                idx = output.find(u'系数')
                output = output[:idx] +  (u'alpha=%.6f，' % lin.alpha_)+ output[idx:]
            if hasattr(lin, 'l1_ratio_'):
                idx = output.find(u'系数')
                output = output[:idx] + (u'l1_ratio=%.6f，' % lin.l1_ratio_) + output[idx:]
            print output,lin.coef_.ravel()

            label = u'%d阶，$R^2$=%.3f' % (j+1,s)
            if hasattr(lin,'l1_ratio_'):
                label += u'，L1 ratio=%.2f' % lin.l1_ratio_
            plt.plot(x_hat,y_hat,color=clrs[j],lw=line_width[j],alpha=0.6,label=label, zorder=j)

            r2, corr_coef = xss(y, model.predict(x))

        plt.legend(loc='upper left')
        plt.grid(True)
        plt.title(titles[i])
        plt.xlabel("X")
        plt.ylabel("Y")
    plt.tight_layout(1,rect=(0,0,1,0.95))
    plt.suptitle(u'多项式拟合曲线比较',fontsize=20)
    plt.show()




    y_max = max(max(tss_list), max(ess_rss_list)) * 1.05
    plt.figure(figsize=(9, 7), facecolor='w')
    t = np.arange(len(tss_list))
    plt.plot(t, tss_list, 'ro-', lw=2, label=u'TSS(Total Sum of Squares)')
    plt.plot(t, ess_list, 'mo-', lw=1, label=u'ESS(Explained Sum of Squares)')
    plt.plot(t, rss_list, 'bo-', lw=1, label=u'RSS(Residual Sum of Squares)')
    plt.plot(t, ess_rss_list, 'go-', lw=2, label=u'ESS+RSS')
    plt.ylim((0, y_max))
    plt.legend(loc='center right')
    plt.xlabel(u'实验：线性回归/Ridge/LASSO/Elastic Net', fontsize=15)
    plt.ylabel(u'XSS值', fontsize=15)
    plt.title(u'总平方和TSS=？', fontsize=18)
    plt.grid(True)
    plt.show()







if __name__ == "__main2__":
    np.random.seed(0)
    N = 9
    x = np.linspace(0, 6, N) + np.random.randn(N)
    x = np.sort(x)
    y = 4*(x**2) + x - 1 + np.random.randn(N)
    x.shape = -1, 1
    y.shape = -1, 1

    print x
    print y
    plt.figure(figsize=(18, 12), facecolor='w')
    plt.grid(True)
    plt.plot(x,y,'ro')
    model = Pipeline([('poly',PolynomialFeatures()),('linear',LinearRegression(fit_intercept=False))])
    model.set_params(poly__degree=5)
    model.fit(x,y.ravel())
    lin = model.get_params('linear')['linear']
    x_hat = np.linspace(x.min(),x.max(),100)
    x_hat.shape = -1,1
    y_hat = model.predict(x_hat)
    plt.plot(x_hat,y_hat,'g-')

    s = model.score(x, y)
    r2, corr_coef = xss(y, model.predict(x))
    print '==========='
    print s
    print '==========='
    print r2
    print '==========='
    print corr_coef

    plt.show()


if __name__ == "__main3__":
    warnings.filterwarnings("ignore")  # ConvergenceWarning
    np.set_printoptions(linewidth=1000)
    N = 9
    print '==========='
    print np.random.randn(N)
    print '==========='
    np.random.seed(10)
    x = np.linspace(0, 6, N) + np.random.randn(N)
    x = np.sort(x)
    np.random.seed(0)
    y = 4*(x**2) - x - 3 + np.random.randn(N)
    x.shape = -1, 1
    y.shape = -1, 1
    print x
    print y
    models = [Pipeline([('poly',PolynomialFeatures()),('linear',LinearRegression(fit_intercept=False))]),
        Pipeline(
            [('poly', PolynomialFeatures()), ('linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
        Pipeline(
            [('poly', PolynomialFeatures()), ('linear', LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
        Pipeline(
            [('poly', PolynomialFeatures()), ('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 40), l1_ratio=[.1,.5,.7,.9,.95,.99,1], fit_intercept=False))])
    ]
    np.set_printoptions(suppress=True)

    plt.figure(figsize=(18,12),facecolor='w')
    d_pool = np.arange(1,N,1)   #阶
    m = d_pool.size
    clrs = [] #颜色
    for c in np.linspace(16711680,255,m):
        clrs.append('#%06x'%c)
    print clrs
    line_width = np.linspace(5, 2, m)
    # print 'x'
    # print x
    # print 'clrs'
    # print clrs
    # print 'line_width'
    # print line_width

    titles = u'线性回归', u'Ridge回归', u'LASSO', u'ElasticNet'
    tss_list = []
    rss_list = []
    ess_list = []
    tss_rss_list = []
    for t in range(4):
        model = models[t]
        plt.subplot(2, 2, t + 1)
        plt.plot(x,y,'ro',ms=10,zorder=N)
        for i, d in enumerate(d_pool):
            model.set_params(poly__degree=d)
            model.fit(x,y.ravel())
            lin = model.get_params('linear')['linear']
            output = u'%s:%d阶，系数为：' % (titles[t],d)
            if hasattr(lin,'alpha_'):
                idx = output.find(u'系数')
                output = output[:idx] + (u'alpha=%.6f，' % lin.alpha_) + output[idx:]
            if hasattr(lin,'l1_ratio_'):    # 根据交叉验证结果，从输入l1_ratio(list)中选择的最优l1_ratio_(float)
                idx = output.find(u'系数')
                output = output[:idx] + (u'l1_ratio=%.6f，' % lin.l1_ratio_) + output[:idx]
            print output, lin.coef_.ravel()
            x_hat = np.linspace(x.min(),x.max(),num=100)
            x_hat.shape = -1, 1
            y_hat = model.predict(x_hat)
            s = model.score(x,y)
            #r2, corr_coef = xss(y, model.predict(x))
            # print 'R2和相关系数：', r2, corr_coef
            # print 'R2：', s, '\n'
            z = N - 1 if (d == 2) else 0
            label = u'%d阶，$R^2$=%.3f' % (d,s)
            if hasattr(lin, 'l1_ratio_'):
                label += u'，L1 ratio=%.2f' % lin.l1_ratio_
            plt.plot(x_hat,y_hat,color=clrs[i],lw=line_width[i], alpha=0.75, label=label,zorder=z)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.title(titles[t], fontsize=18)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.suptitle(u'多项式曲线拟合比较', fontsize=22)
    plt.show()






