#!/usr/bin/python
#  -*- coding:utf-8 -*-

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
    a[a < 0] = 0
    a[a > 255] = 255
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')

if __name__ == "__main__":
    A = Image.open("01.jpg","r")
    output_path = r'./Pic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    # print a.shape
    K = 50
    u_r, sigma_r, v_r = np.linalg.svd(a[ :, :,0])
    # print  u_r.shape
    # print  u_r
    # print '======'
    # print  sigma_r.shape
    # print  sigma_r
    # print '======'
    # print  v_r.shape
    # print  v_r
    u_g, sigma_g, v_g = np.linalg.svd(a[ :, :,1])
    u_b, sigma_b, v_b = np.linalg.svd(a[ :, :,2])
    # plt.figure(figsize=(10,10), facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    # for k in range(1, K+1):
    #     print k
    #     R = restore1(sigma_r, u_r, v_r, k)
    #     G = restore1(sigma_g, u_g, v_g, k)
    #     B = restore1(sigma_b, u_b, v_b, k)
    #     I = np.stack((R, G, B), 2)
    #     Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
    #     if k <= 12:
    #         plt.subplot(3, 4, k)
    #         plt.imshow(I)
    #         plt.axis('off')
    #         plt.title(u'奇异值个数：%d' % k)
    k = 100
    R = restore1(sigma_r, u_r, v_r, k)
    G = restore1(sigma_g, u_g, v_g, k)
    B = restore1(sigma_b, u_b, v_b, k)
    I = np.stack((R, G, B), 2)
    print '============================='
    print I
    print '============================='
    Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
    # plt.subplot(3, 4, 0)
    plt.imshow(I)
    plt.axis('off')
    plt.title(u'奇异值个数：%d' % k)

    plt.suptitle(u'SVD与图像分解', fontsize=20)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.9)
    plt.show()