#!/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = '../datasets/7.iris.data'
    df = pd.read_csv(path,header=None)
    x = df.values[:,:-1]
    y = df.values[:,-1]
    y = LabelEncoder().fit_transform(y)
    print df
    print 'x=\n',x
    print 'y=\n',y
    # 为了可视化取两个特征
    x = x[:,:2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    print y_test

    # 决策树参数估计
    # min_samples_split = 10: 如果该节点包含的样本数大于10，则（有可能）对其分支
    # min_samples_leaf = 10: 如果该节点分之后，得到的每个自己诶单样本数都大于10，则对其分支，否则不进行分支
    model = DecisionTreeClassifier(criterion='gini', max_depth=6)
    model = model.fit(x_train,y_train)
    y_test_hat = model.predict(x_test)

    # 保存
    # dot -Tpng my.dot -o my.png
    # 1.输出
    with open('iris2.dot', 'w') as f:
        tree.export_graphviz(model, out_file=f)
    # 2.给定文件名
    # tree.export_graphviz(model,out_file='iris.dot')

    # 3.输出为pdf格式
    iris_feature_names = 'sepal length', 'sepal width', 'petal length', 'petal width'
    iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_names, class_names=iris_class,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris2.pdf')
    f = open('iris2.png','wb')
    f.write(graph.create_png())
    f.close()

    # 画图
    N, M = 50, 50 #纵横各采样50个值
    x1_min, x1_max = x[:, 0].min(),x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(),x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    # print 't1=\n',t1
    x1, x2 = np.meshgrid(t1, t2)
    # print 'x1=\n',x1
    x_show = np.stack((x1.flat,x2.flat),axis=1)
    # print 'x_show=\n',x_show

    # # 无意义，只是为了凑另外两个维度
    # # 打开该注释前，确保注释掉x = x[:, :2]
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#FFA0A0','#A0ffA0','#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['r','g','b'])
    y_show_hat = model.predict(x_show) # 预测值
    print y_show_hat.shape
    print x_show.shape
    print x1.shape
    y_show_hat = y_show_hat.reshape(x1.shape)   # 使之变为网格形状
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)   # 预测值显示

    plt.scatter(x_test[:,0], x_test[:,1], c=y_test.ravel(), s=120, cmap=cm_dark, marker='*') #测试数据
    plt.scatter(x[:,0], x[:,1], c=y.ravel(),  s=40, cmap=cm_dark) #全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
    plt.show()



    # 训练集上的预测结果
    y_test = y_test.reshape(-1)
    print y_test_hat
    print y_test
    result = (y_test_hat == y_test)   # True则预测正确，False则预测错误
    acc = np.mean(result)
    print '准确度: %.2f%%' % (100 * acc)

    # 过拟合：错误率
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf = clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        err = 1 - np.mean(result)
        err_list.append(err)
        # print d, ' 准确度: %.2f%%' % (100 * err)
        print d, ' 错误率: %.2f%%' % (100 * err)
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()