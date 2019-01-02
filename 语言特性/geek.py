#!/usr/bin/python
# -*- coding: UTF-8 -*-

import re
import os
import numpy as np
from operator import mul  # operator有一些内置函数符


# DocStrings 文档字符串
# 在函数的第一个逻辑行的字符串是这个函数的 文档字符串
# 文档字符串的惯例是一个多行字符串，它的首行以大写字母开始，句号结尾。第二行是空行，从第三行开始是详细的描述。
# 强烈建议你在你的函数中使用文档字符串时遵循这个惯例。
def get_max(a, b):
    """Prints the maximum of two numbers.

    The two values must be integers."""
    if a > b:
        print a
    else:
        print b
    print a if a > b else b


# 解决编码问题
import sys
reload(sys)
sys.setdefaultencoding('utf8')


# 二维数组
mylist = [[0 for j in range(3)] for i in range(4)]  # 4 3 数组  [[]] * n错，是同一引用
mylist[0].append(26)
mylist[1].append(65)
mylist[2].append(535)
mylist[0].append(3)
for ls in mylist:
    ls.sort()
print mylist


# 新增了一种格式化字符串的函数str.format()
print '{:.2f}'.format(321.33345)
print '{0} + {1} + {2} + {1}'.format(1, 2, 3)


# 交换两个数的值
a, b = 1, 2
a, b = b, a
print a, b


# 于任意对象，直接判断其真假
name = 'Tim'
langs = ['AS3', 'Lua', 'C']
info = {'name': 'Tim', 'sex': 'Male', 'age': 23}
if name and langs and info:
    print('All True!')  # All True!


# 字符串反转
s = 'aaaabbbbb'
print s[::-1]


# 列表求和，最大值，最小值，乘积
numList = [1, 2, 3, 4, 5]
sum_ = sum(numList)  # sum = 15
maxNum = max(numList)  # maxNum = 5
minNum = min(numList)  # minNum = 1

prod = reduce(mul, numList)  # prod = 120
print prod


# 列表推导式
print[x * x for x in range(10) if x % 3 == 0]
print {x: x+1 for x in range(5)}  # tuple


# for…else…语句
for x in xrange(1, 5):
    if x == 5:
        print 'find 5'
        break
else:  # 出来for中没有的情况
    print 'can not find 5!'


# 三元符的替代
a = 3
b = 2 if a > 2 else 1


# Enumerate
# 使用enumerate可以一次性将索引和值取出，避免使用索引来取值
# 第二个参数可以调整索引下标的起始位置，默认为0
array = [1, 2, 3, 4, 5]
for i, e in enumerate(array):
    print i, e


# 读取文件
if os.path.exists('a.txt'):
    with open('a.txt') as f:
        for line in f:
            print line
else:
    print 'file not exist!'


# 求最小值的索引
xx = np.mat([[1, 2], [1, 3]])
print np.array(np.where(xx[:, 0] == xx[:, 0].min()))


# lambda语句被用来创建新的函数对象，并且在运行时返回它们
def make_repeater(n):
    return lambda s: s * n
twice = make_repeater(2)  # 创建新的函数对象，并且返回它 绑定了一个匿名函数
print twice("word")


exec 'print "Hello World"'  # 使用exec语句执行包含在字符串中的语句
print eval('2*3')  # eval语句用来计算存储在字符串中的有效Python表达式


mylist = [1]
assert len(mylist) == 1  # 当断言非真时 引发一个AssertionError