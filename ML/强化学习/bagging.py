#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np
import operator
from collections import defaultdict
import os
import matplotlib
import matplotlib.pyplot as plt


def loadData(filename):
    '''
    返回数据列表 和 对应的标号
    '''
    dataList = []
    classLabel = []
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                line = line.strip().split()
                dataList.append([float(line[0]), float(line[1])])
                classLabel.append(float(line[2]))
    else:
        print 'file not exist!'
    return dataList, classLabel


def dataBaggingSamples(dataList, classLabel,p = 0.632):
    '''
    dataList是原始数据集， p是采样比例
    返回采样训练集
    '''
    n = len(dataList)
    m = int(n * p)
    retDatas = []
    retLabels = []
    for i in range(m):
        index = int(np.random.uniform(0, n))
        retDatas.append(dataList[index])
        retLabels.append(classLabel[index])
    return retDatas, retLabels


def classifySDT(dataArr, dimen, threshod, threshodLabel):
    '''
    单层决策树
    由阀值返回类标号
    '''
    retClassLal = np.ones(dataArr.shape[0])
    if threshodLabel == 'lt':
        retClassLal[dataArr[:, dimen] < threshod] = -1
    else:
        retClassLal[dataArr[:,dimen] > threshod] = -1
    return retClassLal


def buildStump(dataArr, classLabel):
    '''
    根据样本的加权计算错误率, 构建最佳单层决策树
    返回单层决策树、错误率、决策树类标号预测
    '''
    m,n = dataArr.shape
    bestStump = {}; minError = np.inf;
    numStep = 10.0
    for i in range(n):  # 对于每个特征
        minVal = dataArr[:,i].min(); maxVal = dataArr[:,i].max() # 得特征的最大最小值
        stepSize = (maxVal - minVal)/numStep
        for j in range(-1, int(numStep)+1):  # 连续值特征，阀值逐步选取
            threshVal = minVal + float(j)*stepSize
            for inequal in ['lt','gt']:  # 对于每个不等号
                pdtClassLabel = classifySDT(dataArr, i, threshVal, inequal)
                errArr = np.ones(m)
                errArr[pdtClassLabel == classLabel] = 0
                wgtError = np.sum(errArr)
                if wgtError < minError:  # 得到最佳决策树
                    minError = wgtError
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['inequal'] = inequal
    return bestStump


def plotWeakClassers(dataArr, classLabel, weakClassers):
    m = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if int(classLabel[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.xlabel('X1')
    plt.ylabel('X2')
    for classer in weakClassers:
        if classer['dim'] == 0:
             plt.axvline(classer["thresh"],linewidth=1)
        else:
            plt.axhline(classer["thresh"],linewidth=1)
    plt.show()


def baggingSD(filename, numT = 6):
    '''
    
    用单层决策树作为基学习器
    '''
    dataList, dataLabel = loadData(filename)
    weakClassers = []
    for i in range(numT):
        ds, dslabel = dataBaggingSamples(dataList, dataLabel)
        bestStump = buildStump(np.array(ds),dslabel)
        weakClassers.append(bestStump)
    plotWeakClassers(np.array(dataList), dataLabel, weakClassers)
    return weakClassers


def baggingClassify(data, weakClassers):
    voteLabel = defaultdict(int)
    for classer in weakClassers:
        prdLabel = classifySDT(np.array([data]), classer['dim'], classer['thresh'], classer['inequal'])
        voteLabel[prdLabel[0]] += 1
    sortedVoteLabel = sorted(voteLabel.iteritems(),
                             key=operator.itemgetter(1), reverse=True)
    return sortedVoteLabel[0][0]


if __name__ == '__main__':
    weakClassers = baggingSD('data1.txt')
    print baggingClassify([1.5, 1.9], weakClassers)
