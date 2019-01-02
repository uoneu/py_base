#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# adaboost算法
#

import numpy as np
import os
import matplotlib.pyplot as plt


def classify_sdt(dataset, dimen, threshod, lgt):
    """
    单层决策树
    由阀值返回类标号
    """
    label = np.ones(dataset.shape[0])
    if lgt == 'lt':
        label[dataset[:, dimen] < threshod] = -1
    else:
        label[dataset[:, dimen] > threshod] = -1
    return label




def stump(dataset, data_label, d):
    """
    根据样本的加权计算错误率, 构建最佳单层决策树
    返回单层决策树、错误率、决策树类标号预测
    """
    m, n = dataset.shape
    bestt_stump = {}; min_error = np.inf; best_label = np.ones(m)
    steps = 10.0
    for i in range(n):  # 对于每个特征
        min_val = dataset[:, i].min(); max_val = dataset[:, i].max()  # 得特征的最大最小值
        step_size = (max_val - min_val)/steps
        for j in range(-1, int(steps)+1):  # 连续值特征，阀值逐步选取
            thresh_val = min_val + float(j) * step_size
            for inequal in ['lt', 'gt']:  # 对于每个不等号
                pdt_label = classify_sdt(dataset, i, thresh_val, inequal)
                err = np.ones(m)
                err[pdt_label == data_label] = 0
                wgt_err = np.sum(d * err)  # 计算加权错误
                if wgt_err < min_error:  # 得到最佳决策树
                    min_error = wgt_err
                    best_label = pdt_label.copy()
                    bestt_stump['dim'] = i
                    bestt_stump['thresh'] = thresh_val
                    bestt_stump['inequal'] = inequal
    return bestt_stump, min_error, best_label


class Adaboost(object):
    """ 提升算法 """
    def __init__(self, train_x, train_y, weak_model=stump, n=30):
        """ nums是弱学习器的个数 """
        self.x = np.array(train_x)
        self.y = np.array(train_y)
        self.weak_model = weak_model
        self.n = n
        self.models = []  # 弱学习器
        self.train()

    def train(self):
        m, n = self.x.shape
        dt = np.ones(m) / m  # 使其成为一个分布，和为1
        prd = np.zeros(m)
        for i in range(self.n):
            stump, error, py = self.weak_model(self.x, self.y, dt)
            alpha = 0.5 * np.log((1.0 - error) / max(error, 1e-16))
            stump['alpha'] = alpha
            self.models.append(stump)  # 将最佳弱决策树加入决策组
            expon = self.y * py * -1.0 * alpha
            dt = dt * np.exp(expon)  # 调整样本权值
            dt = dt / dt.sum()  # 使dt为分布
            prd += alpha * py  # 类估计的累计值
            err_sum = (np.sign(prd) != self.y) * np.ones(m)  # bool值会转为0/1
            if err_sum.sum() == 0:
                break

    def predict(self, data):
        x = np.array(data)
        if len(x.shape) == 1:
            x = np.array([data])
        sums = 0
        for classer in self.models:
            y = classify_sdt(x, classer['dim'], classer['thresh'], classer['inequal'])
            sums += classer['alpha'] * y
        return np.sign(sums)


def load_data():
    data_x, data_y = [], []
    if os.path.exists('data1.txt'):
        with open('data1.txt') as f:
            for line in f:
                line = line.strip().split()
                data_x.append([float(line[0]), float(line[1])])
                data_y.append(float(line[2]))
    else:
        print 'file not exist!'
    return data_x, data_y


def plot_ad(ss):
    x_min, x_max = ss.x[:, 0].min()-0.2, ss.x[:, 0].max()+0.2
    y_min, y_max = ss.x[:, 1].min()-0.2, ss.x[:, 1].max()+0.2
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = ss.predict([x for x in np.c_[xx.ravel(), yy.ravel()]])
    z = np.array(z).reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    sc = ['b' if x == -1 else 'r' for x in ss.y]
    plt.scatter(ss.x[:, 0], ss.x[:, 1], c=sc, s=30)
    plt.show()


# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def load_horse_colic(filename):
    n = len(open(filename).readline().strip().split())
    dataList = []
    dataLabels = []
    fr = file(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineArr[-1] = -1 if float(lineArr[-1]) else 1
        dataArr = []
        for i in range(len(lineArr)-1):
            dataArr.append(float(lineArr[i]))
        dataList.append(dataArr)
        dataLabels.append(float(lineArr[-1]))
    fr.close()
    return np.array(dataList), np.array(dataLabels)


def horse_colic_test():
    x, y = load_horse_colic("horseColicTraining.txt")  # 加载训练数据
    ad = Adaboost(x, y)  # 训练model
    tx, ty = load_horse_colic("horseColicTest.txt")  # 加载测试数据
    py = ad.predict(tx)  # 预测
    print (py != ty).sum() / float(tx.shape[0])

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''


if __name__ == "__main__":
    data_x, data_y = load_data()
    ad = Adaboost(data_x, data_y, stump)
    plot_ad(ad)
    horse_colic_test()









