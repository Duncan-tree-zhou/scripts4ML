# !/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.colors
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def restore_image(cb, cluster, shape):
    row, col, dummy = shape
    image = np.empty((row, col, 3))
    index = 0
    for r in range(row):
        for c in range(col):
            image[r, c] = cb[cluster[index]]
            index += 1
    return image

def show_scatter(a):
    N = 10
    print '原始数据：\n', a

    # 统计频率，density为频率值，edges为坐标系取值
    density, edges = np.histogramdd(a, bins=[N,N,N], range=[(0,1), (0,1), (0,1)])

    # 大小做归一化处理
    density /= density.max()
    x = y = z = np.arange(N)
    xd,yd,zd = np.meshgrid(x, y, z)
    print xd

    # 画图
    fig = plt.figure(1, facecolor='w')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xd, yd, zd, c='r', s=100*density, marker='o', depthshade=True)
    ax.set_xlabel(u'红色分量')
    ax.set_ylabel(u'蓝色分量')
    ax.set_zlabel(u'绿色分量')
    plt.title(u'图像颜色三维频数分布', fontsize=20)

    # 按从高频到低频绘图
    plt.figure(2, facecolor='w')
    den = density[density > 0]
    print np.sort(den)
    den = np.sort(den)[::-1]
    print den
    t = np.arange(len(den))
    plt.plot(t, den, 'r-', t, den, 'go', lw=2)
    plt.title(u'图像颜色频数分布', fontsize=18)
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 做256个像素的聚类压缩
    num_vq = 256
    im = Image.open('../images/01.jpg')
    image = np.array(im).astype(np.float) / 255
    # 不需图片像素点坐标，只需要把每个像素作为一个样本点
    image_v = image.reshape((-1,3))
    print image_v.shape
    model = KMeans(num_vq)
    # show_scatter(image_v)

    N = image_v.shape[0]    #图像像素总数
    # 选择足够多的样本，计算聚类中心
    # 因为像素点太多，kmeans时间复杂度过高，因此采样进行聚类
    idx = np.random.randint(0, N, size=5000)
    image_sample = image_v[idx]
    model.fit(image_sample)
    # model.fit(image_v)
    c = model.predict(image_v)  # 聚类结果
    print '聚类结果：\n', c
    print '聚类中心：\n', model.cluster_centers_

    plt.figure(figsize=(7, 8), facecolor='w')
    plt.subplot(211)
    plt.axis('off')
    plt.title(u'原始图片', fontsize=18)
    plt.imshow(image)
    # plt.savefig('1.png')

    plt.subplot(212)
    vq_image = restore_image(model.cluster_centers_, c, image.shape)
    plt.axis('off')
    plt.title(u'矢量量化后图片：%d色' % num_vq, fontsize=20)
    plt.imshow(vq_image)
    # plt.savefig('2.png')

    # plt.tight_layout(1.2)
    plt.show()

