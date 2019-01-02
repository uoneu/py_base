#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import operator
from collections import defaultdict
import re
import matplotlib
import matplotlib.pyplot as plt

'''
    文本分类主要涉及两方面
        如何构建文本的特征向量，因为文本的单词众多，选择哪些单词作为特征
        利用朴素贝叶斯分类，如何计算各特征的后验概率
    朴素贝叶斯分类是一种惰性学习，
'''


def loadDataSet():
    """
    句子集合、切分好的句子词条
    每个句子的标号
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dadaList):
    '''
    构建词汇表，方便特征向量的构造
    '''
    vocabSet = set([])
    for words in dadaList:
        vocabSet = vocabSet | set(words)
    return list(vocabSet)


def createWords2Vec(vocabList, inputWordList):
    '''
    构建inputWordList词条的特征向量
    初始构建和词汇表长度相同的0向量，inputWordList中的单词在词汇表中出现，置1
    '''
    retVec = [0] * len(vocabList)
    for word in inputWordList:
        try:
            retVec[vocabList.index(word)] = 1  # .incex(xx)返回句子索引
        except ValueError:
            continue
    return retVec


def trainNB0(trainDataArr, trainLabelList):
    '''
    计算每个特征的后验概率以及每个类所占概率
    '''
    m, n = trainDataArr.shape
    p0Num, p1Num = np.ones(n), np.ones(n)
    p0Denom, p1Denom = 2.0, 2.0  # 因为每个属性两种取值 0/1
    pAbuse = np.sum(trainLabelList) / float(m)
    for i in range(m):
        if trainLabelList[i] == 1:
            p1Num += trainDataArr[i]
            p1Denom += 1
        else:
            p0Denom += 1
            p0Num += trainDataArr[i]
    return np.log(p0Num / p0Denom), np.log(p1Num / p1Denom), pAbuse


def classifyNB(dataVecArr, p0VecArr, p1VecArr, pClass):
    '''
    利用模型进行分类
    '''
    p0 = np.sum(dataVecArr * p0VecArr) + np.log(1 - pClass)
    p1 = np.sum(dataVecArr * p1VecArr) + np.log(pClass)
    return 0 if p0 > p1 else 1


def testNB():
    datalist, classVec = loadDataSet()
    vocabList = createVocabList(datalist)
    # print createWords2Vec(vocabList, datalist[0])
    trainList = []
    for sentences in datalist:
        trainList.append(createWords2Vec(vocabList, sentences))
    p0Vec, p1Vec, pAb = trainNB0(np.array(trainList), classVec)
    data1 = createWords2Vec(vocabList, ['worthless', 'good'])
    print classifyNB(data1, p0Vec, p1Vec, pAb)


# 垃圾邮件检测


def textParse(bigString):
    """ 筛选文本单词病转换为小写 """
    strList = re.split(r'\W+', bigString)
    return [s.lower() for s in strList if len(s) > 3]


def spam():
    docList, classList = [], []
    for i in range(1, 26):
        wordList = textParse(open('email//ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 留存交叉验证； 数据的一部分作为训练集， 另一部分作为测试集
    trainIndex = range(50)
    taianDate, trainLabel = [], []
    for i in range(40):
        index = int(np.random.uniform(0, len(trainIndex)))
        taianDate.append(docList[index])
        trainLabel.append(classList[index])
        del(trainIndex[index])
    trainDataArr = []
    # 训练模型，事先求出各特征的概率
    for sentences in taianDate:
        trainDataArr.append(createWords2Vec(vocabList, sentences))
    p0Vec, p1Vec, pAb = trainNB0(np.array(trainDataArr), trainLabel)
    # 测试，查看错误率
    errorSum = 0.0
    for index in trainIndex:
        testVec = createWords2Vec(vocabList, docList[index])
        if classifyNB(testVec, p0Vec, p1Vec, pAb) != classList[index]:
            errorSum += 1
            print docList[index]
    print errorSum


if __name__ == '__main__':
    # testNB()
    print textParse('100% herbal, 100% Natural, 100% Safe')
    spam()
