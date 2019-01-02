#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# logist回归
#

import numpy as np
import operator
from collections import defaultdict
import os
import matplotlib
import matplotlib.pyplot as plt




def loadDataSet():
    '''
    加载数据，构建属性和类标号列表，并返回
    '''
    labelMat = []
    dataMat = []
    fp = file("test1.txt")
    for line in fp.readlines():
        lineArr = line.strip().split()
        labelMat.append(int(lineArr[2]))
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    return dataMat, labelMat


def Sigmoid(intX):
    '''
    Sigmoid 函数
    '''
    return 1.0 / (1 + np.exp(-intX))  # numpy的n维数组对象、可以对这种数组每个元素执行同一种运算


def gradAscent(dataMatIn, classLabel):
    '''
    梯度下降算法
    使用numpy中矩阵
    '''
    np.set_printoptions(suppress=True)
    dataMatrix = np.mat(dataMatIn)  # 构建矩阵
    labelMatrix = np.mat(classLabel).transpose()  # 矩阵的转置
    m, n = dataMatrix.shape  # 等价于shape(dataMatrix)
    alpha = 0.001  # 步长
    maxCycle = 500  # 迭代次数
    weights = np.ones((n, 1))  # 系数初始化为1
    for k in range(maxCycle):
        h = Sigmoid(dataMatrix * weights)  # 将返回一个列向量
        error = labelMatrix - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def gradAscent_1(dataMatIn, classLabel):
    '''
    梯度下降算法
    参数dataMatIn, classLabel是列表
    不使用矩阵
    '''
    np.set_printoptions(suppress=True)
    dataMatrix = np.array(dataMatIn)
    labelMatrix = np.array(classLabel).reshape(len(classLabel), 1)  # 矩阵的转置
    m, n = dataMatrix.shape  # 等价于shape(dataMatrix)
    alpha = 0.001  # 步长
    maxCycle = 500000  # 迭代次数
    weights = np.ones((n, 1))  # 系数初始化为1
    histWgt = []
    for k in range(maxCycle):
        h = Sigmoid(dataMatrix.dot(weights))  # 将返回一个列向量
        error = labelMatrix - h
        weights = weights + alpha * dataMatrix.T.dot(error)
        histWgt.append(weights)
    return weights, histWgt


def gradAscent_2(dataMatIn, classLabel):
    '''
    随机梯度下降算法
    参数dataMatIn, classLabel是列表
    不使用矩阵
    '''
    np.set_printoptions(suppress=True)
    dataMatIn = np.array(dataMatIn)  # 先把list抓换成numpy的数组
    m, n = np.shape(dataMatIn)
    alpha = 0.5  # 步长
    weights = np.ones(n)
    histWgt = []
    for k in range(200):
        for i in range(m):
            h = Sigmoid(np.sum(dataMatIn[i] * weights))
            error = classLabel[i] - h
            weights = weights + alpha * error * dataMatIn[i]
            histWgt.append(weights)
    return weights, histWgt


def gradAscent_3(dataMatIn, classLabel, numIter=150):
    '''
    随机梯度下降算法
    参数dataMatIn, classLabel是列表
    不使用矩阵
    '''
    np.set_printoptions(suppress=True)
    dataMatIn = np.array(dataMatIn)  # 先把list抓换成numpy的数组
    m, n = np.shape(dataMatIn)
    weights = np.ones(n)
    histWgt = []
    for i in range(numIter):
        dataSetIndex = range(m)
        for j in range(m):
            alpha = 4 / (i + j + 1.0) + 0.01
            randIndex = int(np.random.uniform(0, len(dataSetIndex)))  # 均匀分布
            h = Sigmoid(np.sum(dataMatIn[randIndex] * weights))
            error = classLabel[randIndex] - h
            weights = weights + alpha * error * dataMatIn[randIndex]
            del(dataSetIndex[randIndex])
            histWgt.append(weights)
    return weights, histWgt

#-------------------------------------


def plotBestFir(weights):
    '''
    画出决策边界
    '''
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    m = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)  # 类似range, 但返回的是ndarray
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def iterPlot(histWgt):
    '''
    查看随着迭代次数系数的变化
    histWgt 是列表
    '''
    histWgtArr = np.array(histWgt)
    fg = plt.figure()
    ax1 = fg.add_subplot(311)
    ax2 = fg.add_subplot(312)
    ax3 = fg.add_subplot(313)
    x = np.arange(histWgtArr.shape[0])
    ax1.plot(x, histWgtArr[:, 0])
    ax1.set_ylabel('X0')
    ax2.plot(x, histWgtArr[:, 1])
    ax2.set_ylabel('X1')
    ax3.plot(x, histWgtArr[:, 2])
    ax3.set_ylabel('X2')
    plt.show()
#-------------------------------------------------------------------------


def classifyVector(intX, weights):
    prob = Sigmoid(np.sum(intX * weights))
    return 1 if prob > 0.5 else 0


def colicTest():
    frTrain = file('horseColicTraining.txt')
    trainSet = []
    trainLabel = []
    for line in frTrain.readlines():
        curLine = line.strip().split()
        lineArr = [1.0]
        for i in range(21):
            lineArr.append(float(curLine[i]))
        trainSet.append(lineArr)
        trainLabel.append(float(curLine[21]))
    frTrain.close()
    trainWgt, histWgt = gradAscent_3(trainSet, trainLabel)

    frTest = file('horseColicTest.txt')
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        curLine = line.strip().split()
        lineArr = [1.0]
        for i in range(21):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(lineArr, trainWgt)) != int(curLine[21]):
            errorCount += 1
    frTest.close()

    errorRate = float(errorCount) / numTestVec
    return errorRate


def mulTest():
    numTest = 10
    errortSum = 0.0
    for i in range(numTest):
        errortSum += colicTest()
    print errortSum / numTest


if __name__ == '__main__':
    dataMatIn, classLabel = loadDataSet()
    weights, histWgt = gradAscent_3(dataMatIn, classLabel)
    print weights
    # plotBestFir(weights)  # getA 矩阵转数组
    iterPlot(histWgt)

    # print weights.shape
    #mulTest()
