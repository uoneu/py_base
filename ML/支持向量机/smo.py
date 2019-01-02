#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# SVM模型
# 序列优化算法smo
#


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class SvmModel(object):
    """ SVM模型 """
    def __init__(self, train_data, train_label, C=0.6, kTup=('lin', ), tol=0.001, maxIter = 600):
        self.m, self.n = np.shape(train_data)
        self.data = np.mat(train_data)
        self.label = train_label
        self.c = C
        self.tol = tol
        self.b = 0.0
        self.kTup = kTup
        self.maxIter = maxIter
        self.alpha = np.mat(np.zeros(self.m))
        self.eCache = np.mat(-self.label, dtype=np.float64)
        # self.w = None
        self.k = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = self.kernel(self.data, self.data[i])
        self.mainRoutine() # 用SMO算法求解
        self.svnInd = np.nonzero(self.alpha.A > 0)[1]
        self.svx = self.data[self.svnInd]   # 支持向量

    def kernel(self, x, y):
        """ 核函数定义 """
        m, n = np.shape(x)
        ker = np.mat(np.zeros((m, 1)))
        if self.kTup[0] == 'lin':
            y = np.mat(y)
            ker = x * y.T
        elif self.kTup[0] == 'rbf':
            for i in range(m):
                z = x[i, :] - y
                ker[i] = z * z.T
            ker = np.exp(ker / (-2 * self.kTup[1] ** 2))
        return ker

    def selectJ(self, i, ei):
        """ 启发式选择第二个变量 """
        maxK, maxE, ej = i, 0, 0
        for k in range(self.m):
            if k == i or not (0 < self.alpha[0, k] < self.c):  # 选择非边界点， 这点很重要！
                continue
            absE = abs(self.eCache[0, i] - self.eCache[0, k])
            if maxE < absE:
                maxK, maxE = k, absE
        return maxK

    def clipAlpha(self, aj, L, H):
        """ 剪辑后aj的解 """
        if aj > H:
            return H
        elif aj < L:
            return L
        else:
            return aj

    def mainRoutine(self):
        itenum, alphaChanged, entireSet = 0, 0, True

        while (alphaChanged > 0 or entireSet) and itenum <= self.maxIter:
            alphaChanged = 0
            if entireSet:  # 在所有数据集上测试
                for i in range(self.m):
                    alphaChanged += self.examineExample(i)
                itenum += 1
            else:  # 在非边界集上测试
                nonBoundIs = np.nonzero((0 != self.alpha.A) * (self.alpha.A != self.c))[1]
                # print nonBoundIs
                for i in nonBoundIs:
                    alphaChanged += self.examineExample(i)
                itenum += 1
            if entireSet:
                entireSet = False
            elif alphaChanged == 0:
                entireSet = True

            itenum += 1

    def examineExample(self, i):
        Ei = self.eCache[0, i]
        r1 = self.label[i] * Ei
        if (r1 < -self.tol and self.alpha[0, i] < self.c) or (r1 > self.tol and self.alpha[0, i] > 0):
            nonBoundIs = np.nonzero((0 < self.alpha.A) * (self.alpha.A < self.c))[1]
            if len(nonBoundIs) > 1:
                j = self.selectJ(i, Ei)
                if self.takeStep(i, j):
                    return 1

            np.random.shuffle(nonBoundIs) # 构建随机访问序列
            # print nonBoundIs
            for j in nonBoundIs:
                # print nonBoundIs
                if self.takeStep(i, j):
                    return 1

            randSq = np.arange(self.m)
            np.random.shuffle(randSq)
            for j in randSq:
                if self.takeStep(i, j):
                    return 1
        return 0

    def calE(self, k):
        """ 计算第k个样本的误差"""
        # fk = np.mat(ss.alpha.A * ss.label) * (ss.data * ss.data[k,:].T) + ss.b
        fk = np.mat(self.alpha.A * self.label) * self.k[:, k] + self.b
        # print fk.shape
        return fk[0, 0] - float(self.label[k])

    def updataEk2cache(self, k):
        """ 更新误差缓存 """
        ek = self.calE(k)
        self.eCache[0, k] = ek

    def takeStep(self, i, j):
        if i == j:
            return 0

        Ei = self.eCache[0, i]
        Ej = self.eCache[0, j]
        tg = self.k[i, i] + self.k[j, j] - 2 * self.k[i, j]
        if tg <= 0:
            return 0
        aiOld = self.alpha[0, i]
        ajOld = self.alpha[0, j]
        ajNewUnc = ajOld + self.label[j] * (Ei - Ej) / tg
        if self.label[i] == self.label[j]:
            L = max(0, ajOld + aiOld - self.c)
            H = min(self.c, ajOld + aiOld)
        else:
            L = max(0, ajOld - aiOld)
            H = min(self.c, self.c + ajOld - aiOld)
        if L == H:
            return 0
        aj = self.clipAlpha(ajNewUnc, L, H)
        if abs(aj - ajOld) < 0.00001:
            return 0
        ai = aiOld + self.label[i] * self.label[j] * (ajOld - aj)
        b1 = -Ei - self.label[i] * self.k[i, i] * (ai - aiOld) - self.label[j] * self.k[i, j] * (aj - ajOld) + self.b
        b2 = -Ej - self.label[i] * self.k[i, j] * (ai - aiOld) - self.label[j] * self.k[j, j] * (aj - ajOld) + self.b
        if 0 < ai < self.c:
            self.b = b1
        elif 0 < aj < self.c:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        self.alpha[0, i] = ai
        self.alpha[0, j] = aj

        for i in range(self.m):
            self.updataEk2cache(i)

        return 1

    def predict(self, data):
        """ 用模型进行分类 """
        cvx = self.kernel(self.svx, data)
        return np.sign(np.mat(self.label[self.svnInd] * self.alpha[0, self.svnInd].A) * cvx + self.b)[0, 0]

    def __del__(self):
        #del self.data, self.eCache, self.k
        pass


def loadData(filename):
    '''' 读取数据 '''
    dataLs, labelLs = [], []
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                lineLs = line.strip().split('\t')
                dataLs.append(map(lambda x: float(x), lineLs[0:-1]))

                labelLs.append(int(float(lineLs[-1])))
            f.close()
        return dataLs, labelLs
    else:
        print 'file not exist!'


def plotSVM(ss):
    x_min, x_max = ss.data[:, 0].min()-0.2, ss.data[:, 0].max()+0.2
    y_min, y_max = ss.data[:, 1].min()-0.2, ss.data[:, 1].max()+0.2
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = [ss.predict(x) for x in np.c_[xx.ravel(), yy.ravel()]]
    z = np.array(z).reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    sc = ['blue' if x == -1 else 'red' for x in ss.label]
    plt.scatter(ss.data.A[:, 0], ss.data.A[:, 1], c=sc, s=30)
    '''
    for i in np.nonzero((0.0000001<ss.alpha.A)*(ss.alpha.A<ss.c))[1]:
        cord = (ss.data[i,0], ss.data[i, 1])
        circle = Circle(cord, 0.4, facecolor='None', edgecolor='red', linewidth=3, alpha=1)
        #plt.add_patch(circle)
    '''
    plt.show()


def t_rbf():
    dataLs, labelLs = loadData('testSetRBF.txt')  # 加载数据
    sm = SvmModel(np.array(dataLs), np.array(labelLs), C=200, kTup=('rbf', 1))  # 训练模型
    plotSVM(sm)


def t_lin():
    dataLs, labelLs = loadData('testSet.txt')
    sm = SvmModel(np.array(dataLs), np.array(labelLs))  # 训练模型
    plotSVM(sm)


if __name__ == "__main__":
    # np.set_printoptions(suppress=True) #显示10进制
    ##t_lin()
    t_rbf()

