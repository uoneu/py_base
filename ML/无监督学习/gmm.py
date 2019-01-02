#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# 高斯混合模型
# EM算法
#

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import time
import copy


class GMMs(object):
    """ 高斯混合模型 """
    def __init__(self, train_data, k, max_itr=500):
        """ 高斯混合成分个数k, 训练训练轮数max_itr """
        np.random.seed(seed=int(time.time()))
        self.data = train_data
        self.k = k
        self.max_itr = max_itr
        rs = np.random.choice(train_data.shape[0], k, replace=False)
        self.means = copy.deepcopy(train_data[rs])  # 随机选择k个点作均值向量
        self.alpha = np.array([1.0 / k] * k)  # 混合系数
        cov = np.cov(train_data.T)
        self.covs = np.array([cov] * k)  # 协方差矩阵采用所有数据的协方差矩阵为初始值
        print self.means
        self.m, self.n = train_data.shape
        print self.alpha[0]
        self.em()

    def em(self):
        """ EM算法 """
        #r = np.array([[0.0 for i in range(self.k)] for j in range(self.m)]) # m * k
        r = np.array([[0.0] * self.k] * self.m)
        # print r[0, 0]
        itr_num = 0
        while itr_num < self.max_itr:
            # 算法的E步，计算xj属于各个分布的后验概率
            for j in range(self.m):
                for i in range(self.k):
                    z = np.sum([self.alpha[k] * multivariate_normal.pdf(self.data[j], self.means[k], self.covs[k]) for k in range(self.k)])
                    r[j, i] = self.alpha[i] * multivariate_normal.pdf(self.data[j], self.means[i], self.covs[i]) / z

            # 算法的M步，计算似然期望的最大值
            for i in range(self.k):
                # print self.means[i]
                rj_sum = np.sum(r[:, i])
                self.means[i] = np.sum(self.data * r[:, i].reshape(self.m, 1), axis=0) / rj_sum
                # self.means[i] = np.sum(np.array([self.data[j] * r[j, i] for j in range(self.m)]), axis=0)
                # print self.means[i]
                # print self.data[i]

                self.covs[i] = np.sum([r[j, i] * np.mat(self.data[j] - self.means[i]).T * np.mat(self.data[j] - self.means[i]) for j in range(self.m)], axis=0) / rj_sum
                # print self.covs[i]
                #exit()
                self.alpha[i] = rj_sum / self.m

            itr_num += 1
        print '---------------'
        print self.covs
        print self.means
        print self.alpha
        print type(self.covs[0])


def show_model():
    np.random.seed(seed=99)

    # make some data up
    mean1 = [0, 2]
    mean2 = [-1, -1]
    mean3 = [2, -2]
    cov1 = [[1.0, 0.0], [0.0, 0.5]]
    cov2 = [[1.0, 0.0], [0.0, 1.0]]
    cov3 = [[0.5, 0.0], [0.0, 0.5]]

    # create some points
    x1 = np.random.multivariate_normal(mean1, cov1, 200)
    x2 = np.random.multivariate_normal(mean2, cov2, 200)
    x3 = np.random.multivariate_normal(mean3, cov3, 200)
    x = np.vstack((x1, x2, x3))

    np.random.shuffle(x)  # 打乱数据

    gmms = GMMs(x, 3, 50)

    # 定义figure
    fig = plt.figure()
    # 将figure变为3d
    ax = Axes3D(fig)

    x = np.arange(-4, 4, 0.2)
    y = np.arange(-4, 4, 0.2)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(gmms.means[0], gmms.covs[0])
    rv1 = multivariate_normal(gmms.means[1], gmms.covs[1])
    rv2 = multivariate_normal(gmms.means[2], gmms.covs[2])

    # 绘制3D曲面
    ax.plot_surface(X, Y, gmms.alpha[0]*rv.pdf(pos) + gmms.alpha[1]*rv1.pdf(pos) + gmms.alpha[2]*rv2.pdf(pos), rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # 绘制从3D曲面到底部的投影
    ax.contour(X, Y, gmms.alpha[0]*rv.pdf(pos) + gmms.alpha[1]*rv1.pdf(pos) + gmms.alpha[2]*rv2.pdf(pos), zdim='z', offset=-0.05, cmap='rainbow')

    # 设置z轴的维度
    ax.set_zlim(-0.05, 0.15)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # 显示10进制
    show_model()




