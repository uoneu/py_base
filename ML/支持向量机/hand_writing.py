#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    SVM 手写字体识别 
    测试smo、核函数
"""
import os
import re
import numpy as np

import smo


def dealHandfile():
    """ 去除非0和9的样本, 此SVM 仅识别两类 0 9"""
    tainFileList = os.listdir('trainingDigits')
    for filename in tainFileList:
        if re.match(r'[2-8]_.+', filename):
            os.remove('trainingDigits\\' + filename)

    testFileList = os.listdir('testDigits')
    for filename in testFileList:
        if re.match(r'[2-8]_.+', filename):
            os.remove('testDigits\\' + filename)


def createVectorFromImage(filename):
    """ 将图像转换为向量 """
    returnVect = np.zeros((1024))
    if os.path.exists(filename):
        with open(filename) as fp:
            fileLines = fp.readlines()
            for i in xrange(32):
                for j in range(32):
                    returnVect[i*32+j] = int(fileLines[i][j])
        fp.close()
        return returnVect
    else:
        print '%s not exist!' % filename


def handWritingTest():
    # 训练模型
    imageData, imageLabel = [], []
    tainFileList = os.listdir('trainingDigits')

    for filename in tainFileList:
        imageData.append(createVectorFromImage('trainingDigits\\' + filename))
        imageLabel.append(-1) if filename[0]=='1' else imageLabel.append(1)

    sm = smo.SvmModel(np.array(imageData), np.array(imageLabel), C=200, kTup=('rbf', 10), maxIter=1000)

    # 测试误差( 泛化能力 )
    testFileList = os.listdir('testDigits')
    i = 0
    for filename in testFileList:
        imageVec = np.array(createVectorFromImage('testDigits\\' + filename))
        prd = 1 if sm.predict(imageVec) == -1 else 9
        if prd != int(filename[0]):
            print filename


if __name__ == '__main__':
    # dealHandfile()
    handWritingTest()




