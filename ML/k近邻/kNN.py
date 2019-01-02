#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# K近邻居算法
#

import numpy as np
import operator
from collections import defaultdict
import os
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classfy0(inx, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape 数组大小 0是第0轴 （行）
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataSet
    # print diffMat
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    # print sqDistance
    distance = sqDistance**0.5
    sortDictIndex = distance.argsort()  # 得到数组值从小到大的索引值列表
    # print sortDictIndex
    # classCount = {}
    classCount = defaultdict(int)

    for i in range(k):
        voteLabel = labels[sortDictIndex[i]]
        # D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
        # classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        classCount[voteLabel] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    # print sortedClassCount
    return sortedClassCount[0][0]


def file2Matix(filename):
    if os.path.exists(filename):
        with open(filename) as fr:
            arrayOLines = fr.readlines()
            numberOfLines = len(arrayOLines)
            # 先初始化数组， 方便 returnMat[index, :] = listFromLine[0:3]数据类型转换
            returnMat = np.zeros((numberOfLines, 3))
            classLabeLVector = []
            index = 0
            for line in arrayOLines:
                line = line.strip()
                listFromLine = line.split('\t')
                returnMat[index, :] = listFromLine[0:3]  # [0,1,2] 前者是数字，后者数据的类型是字符串
                classLabeLVector.append(int(listFromLine[-1]))
                index += 1
            return returnMat, classLabeLVector

    else:
        print '%s not exist!' % filename


def autoNorm(dataSet):
    np.set_printoptions(suppress=True) #显示10进制
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataset = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataset = dataSet - np.tile(minVals, (m, 1))
    normDataset = normDataset / np.tile(ranges, (m, 1))
    return normDataset, ranges, minVals


def datingClassTest():
    '''
    模型测试
    '''
    hoRatio = 0.10
    dataMat, classLabeLVector = file2Matix('datingTestSet2.txt')
    normDataMat, ranges, minVal = autoNorm(dataMat)
    m = normDataMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifyResult = classfy0(normDataMat[i], normDataMat[
                                  numTestVecs:m], classLabeLVector[numTestVecs:m], 4)
        if (classifyResult != classLabeLVector[i]):
            errorCount += 1
    print errorCount
    print "the total error rate is : %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    print('Hi:')
    print('please input the people charactor')
    flyMils = float(raw_input("-->flying mils:"))
    playGameTime = float(raw_input("-->play game time:"))
    iceCream = float(raw_input("-->ice cream consum:"))

    inarr = np.array([flyMils, playGameTime, iceCream])
    dataMat, classLabeLVector = file2Matix('datingTestSet2.txt')
    normDataMat, ranges, minVal = autoNorm(dataMat)
    classifyResult = classfy0(
        (inarr - minVal) / ranges, normDataMat, classLabeLVector, 4)

    result = ["didn't Like", 'small Doses', 'large Doses']
    print "the person maybe you like : %s!" % result[classifyResult - 1]






#---------------------------------------------------------------------------------------------------------

def createVectorFromImage(filenmae):
    returnVect = np.zeros((1,1024))
    if os.path.exists(filenmae):
        with open(filenmae) as fp:
            fileLines = fp.readlines()
            for i in xrange(32):
                for j in range(32):
                    returnVect[0,i*32+j] = int(fileLines[i][j])
        fp.close()
        return returnVect
    else:
        print '%s not exist!' % filename



def handWritingTRest():
    trainFileList = os.listdir('trainingDigits')
    hwLabel = []
    imageDataSet = np.zeros((len(trainFileList), 1024))
    for i, filename in  enumerate(trainFileList):
        hwLabel.append(filename[0])
        imageDataSet[i,:] = createVectorFromImage('trainingDigits/'+filename) #createVectorFromImage('trainingDigits/%s' %filename)

    testFileList = os.listdir('testDigits')
    for filename in testFileList:
        testFileVct = createVectorFromImage('testDigits/'+filename)
        classResult = classfy0(testFileVct, imageDataSet, hwLabel, 3)
        if classResult !=  filename[0]:
            print '%s -> %s error:%s '%(filename, filename[0], classResult)
    #classfy0(,imageDataSet, hwLabel, 3)




if __name__ == '__main__':
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(returnMat[:,1], returnMat[:,2], s=15.0*np.array(classLabeLVector), c=15.0*np.array(classLabeLVector))
    plt.show()
    """
    #classifyPerson() #约会网站识别
    #print createVectorFromImage('0_0.txt')[0][0:31]
    handWritingTRest()


"""
from collections import defaultdict
arr = [1, 1, 1, 5, 1, 1, 1, 2]
count = defaultdict(int)
for num in arr:
    count[num] += 1
print count
"""
