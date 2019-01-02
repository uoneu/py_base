#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
np的数组切片是在原有的视图上，会对原始数据操作
"""

import numpy as np


def part_1():
    np.set_printoptions(suppress=True)  # 显示10进制

    data1 = ['2', '3']
    arr = np.zeros(2)  # 类型以确定
    print arr.dtype
    arr[:] = data1[:]  # 类型转换
    print arr
    print arr.dtype  # 数组自动推断一个合适的数据类型

    print np.zeros(10)
    print np.zeros((3, 6))
    print np.arange(15)
    print np.arange(-5, 5, 1)

    # 数组运算 是矢量运算，运算应用到元素级，注意和矩阵的区别
    arr1 = np.array([1, 2, 3])
    print arr1 * arr1  # 【1 4 9】
    print arr1**0.5  # 根号
    print arr1 - arr1
    print arr1 * 10  # [10 20 30]

    print arr1.shape[0]  # 数组大小
    np.shape(arr1)  # 等价于arr1.shape  但对于python list 只能用np.shape()
    print np.tile(arr1, 2)  # 数组内元素重复2
    print np.tile(arr1, (3, 2))  # 先数组内元素重复2， 然后数组再重复3
    print arr1.sum(axis=0)  # 理解轴的含义

    print
    # 排序
    # argsort函数返回的是数组值从小到大的索引值
    print arr.argsort()

    print
    # 利用数组进行数据处理、矢量化、批处理
    xs = np.arange(4)
    print np.square(xs)  # [0 1 4 9]
    print xs.shape
    print xs.reshape(4, 1)

    print
    # 广播 广播原则：末尾的维度相等或其中的一方长度是1
    arr = np.random.randn(4, 3)
    print arr
    demead = arr - arr.mean(0)
    print demead.mean(0)

    # matrix矩阵
    # 在numpy中的特殊类型，是作为array的子类出现


def part_2():
    print np.all([True, False, False, False])  # 与
    print np.any([True, False, False, False])  # 或
    print np.random.randint(0, 2, (3, ))  # 随机产生0, 1数组  [ np中的随机函数random ]

if __name__ == '__main__':
    #part_2()
    a = range(5)
    print np.random.sample(a, 2)



