#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# 线性回归
# LMS 均方误差算法
# 梯度下降算法
#

import numpy as np
import matplotlib.pyplot as plt
import time


def load_data():
    dataset = []
    f = open('a.txt', 'r')
    for line in f.readlines():
        x = line.strip().split()
        dataset.append([float(x[0]), float(x[1]), float(x[2])])
    f.close()
    return np.array(dataset)


class LR(object):
    """ 线性回归 """
    def __init__(self, dataset, alpha=0.01, max_it=500):
        """ alpha学习率， max_it最大迭代轮数 """
        self.data = np.array(dataset)
        self.alpha = alpha
        self.max_it = max_it
        self.m, self.n = self.data.shape
        self.w = np.ones(self.n-1)
        self.sgd()

    def sgd(self):
        """ 随机梯度下降算法 """
        np.random.seed(seed=int(time.time()))
        a = range(self.m)
        for itr in range(self.max_it):
            rs = np.random.choice(a, self.m, replace=False) # 构造随机访问序列
            for i in rs:
                error = self.data[i, 2] - np.sum(self.data[i, 0:2] * self.w)
                self.w += self.data[i, 0:2]*self.alpha * error
                #self.w += self.alpha * error/self.data[i, 0:2]


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    dataset = load_data()
    lr = LR(dataset, max_it=500)
    # 显示模型
    plt.scatter(dataset[:, 1], dataset[:, 2], color='b', s=10)
    x = np.linspace(0, np.max(lr.data[:,1]), 2)
    y = x*lr.w[1] + lr.w[0]
    plt.plot(x, y, color='r',  label="Linear Regression", linewidth=2)
    plt.legend()  # 显示图示
    plt.show()