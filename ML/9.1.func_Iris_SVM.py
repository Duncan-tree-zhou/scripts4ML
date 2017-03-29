# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm
from sklearn.model_selection import train_test_split


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

# 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print tip + '正确率：', np.mean(acc)

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    path = '../datasets/7.iris.data'
    # 读取花萼长度和花萼宽度
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})
    x,y = np.split(data,(4,),axis=1)
    x = x[:,(0,1)]
    # 创建并训练模型
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, random_state=7)
    model = svm.SVC(C=10, kernel='rbf', gamma=10, decision_function_shape='ovr')
    # model = svm.SVC(C=10, kernel='linear', gamma=10, decision_function_shape='ovr')
    model.fit(x_train,y_train.ravel())
    # 计算分数
    y_hat = model.predict(x_train)
    print 'train score(accuracy): ',model.score(x_train,y_train)
    y_hat_t = model.predict(x_test)
    print 'test score(accuracy): ', model.score(x_test,y_test)

    # 画图
    N, M = 500,500
    x1_min, x1_max = x[:,0].min(), x[:,0].max()
    x2_min, x2_max = x[:,1].min(), x[:,1].max()
    # 生成采样点
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    x1,x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    x_grid = np.stack((x1.flat,x2.flat),axis=1)
    y_grid = model.predict(x_grid)
    y_grid = y_grid.reshape(x1.shape)
    plt.pcolormesh(x1, x2, y_grid, cmap=cm_light) # 模型预测区域
    plt.scatter(x[:, 0],x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark) # 训练样本
    plt.scatter(x_test[:,0],x_test[:,1], s=120, facecolors='none',zorder=10)    # 测试样本
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
    plt.grid()
    plt.show()








