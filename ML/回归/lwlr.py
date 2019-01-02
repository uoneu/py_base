#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# 局部加权线性回归
# 牛顿法
#
import numpy as np
import time
import matplotlib.pyplot as plt


def load_data():
    """ 加载数据 """
    dataset = []
    f = open('a.txt', 'r')
    for line in f.readlines():
        x = line.strip().split()
        dataset.append([float(x[0]), float(x[1]), float(x[2])])
    f.close()
    return dataset


class LWR(object):
    """
    局部加权线性回归
    随机梯度
    """
    def __init__(self, x_train, tau=0.04):  # 0.04合适
        self.x = np.array(x_train)
        self.m, self.n = self.x.shape
        self.tau = tau

    def predict(self, x, alpha=0.01, max_it=500):
        np.random.seed(seed=int(time.time()))
        theta = np.zeros(self.n - 1)
        w = np.exp(-(x - self.x[:, 1])**2 / (2*self.tau**2))
        a = range(self.m)
        for itr in range(max_it):
            rs = np.random.choice(a, self.m, replace=False)  # 构造随机访问序列
            for i in rs:
                error = self.x[i, 2] - np.sum(self.x[i, 0:2] * theta)
                if np.abs(error) < 1e-6:
                    #print '---', itr
                    return x * theta[1] + theta[0]
                theta += w[i]*self.x[i, 0:2] * alpha * error
        return x * theta[1] + theta[0]


if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # 显示10进制
    x = load_data()
    lwr = LWR(x[0:100])
    plt.scatter(lwr.x[:, 1], lwr.x[:, 2], color='b', s=10)
    x1 = np.linspace(0, np.max(lwr.x[:, 1]), 100)
    z = [lwr.predict(x) for x in x1]
    plt.scatter(x1, z, color='r', s=11)
    plt.show()