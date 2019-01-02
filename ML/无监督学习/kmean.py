#!/usr/bin/python
# coding=utf-8
#

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
import time
from collections import Counter


def load_data(filename):
    """ 加载数据 """
    data_ls = []
    with open(filename) as flines:
        for data in flines:
            temp = map(lambda x: float(x), data.strip().split('\t'))
            data_ls.append(temp)
    return data_ls


def plot_data(km):
    """ 画出数据分布 """
    x_min, x_max = km.data[:, 0].min() - .5, km.data[:, 0].max() + .5
    y_min, y_max = km.data[:, 1].min() - .5, km.data[:, 1].max() + .5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #print np.c_[[1,2,3], [2,3, 4]]
    z = [ km.predict(x) for x in np.c_[xx.ravel(), yy.ravel()]]
    z = np.array(z).reshape(xx.shape)

    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)

    for data in km.cs:
        plt.scatter(km.data[data][:, 0], km.data[data][:, 1], s=20, marker='o')
    plt.scatter(km.us[:, 0], km.us[:, 1], s=90, c='black', marker='X')
    plt.show()


class KMeans(object):
    """ kmeans聚类分析 """
    def __init__(self, dataset, k, max_itr=1000):
        np.random.seed(seed=int(time.time()))
        self.data = np.array(dataset)
        self.k = k
        self.max_itr = max_itr
        self.m, self.n = np.shape(self.data)
        rs = np.random.choice(range(self.m), k, replace=False)
        self.us = self.data[rs]  # 随机选择k个样本做簇中心
        self.cs = [[] for i in range(k)] # self.cs = [[]]*4 错,是对同一列表引用4次
        self.kmeas()

    def kmeas(self):
        itr, flag = 0, True
        while itr < self.max_itr and flag:
            self.cs = [[] for i in range(self.k)]
            flag = False
            # 计算每个样本所属的族
            for i in range(self.m):
                mind, minj = float("inf"), 0
                for j in range(self.k):
                    dij = np.sqrt(np.sum((self.data[i] - self.us[j]) ** 2))
                    if mind > dij:
                        mind, minj = dij, j
                self.cs[minj].append(i)

            # 更新族中心
            for i in range(self.k):
                # newCi = list(np.sum(np.array(dataLs)[cs[i]], axis=0) / len(cs[i]))
                newCi = np.mean(self.data[self.cs[i]], axis=0)
                if np.all(self.us[i] != newCi):
                    self.us[i] = newCi
                    flag = True
            itr += 1

    def predict(self, data):
        mind, minj = float("inf"), 0
        for j in range(self.k):
            dij = np.sqrt(np.sum((data - self.us[j]) ** 2))
            if mind > dij:
                mind, minj = dij, j
        return minj


if __name__ == '__main__':
    dataLs = load_data('testSet.txt')
    km = KMeans(dataLs, 4)
    plot_data(km)



