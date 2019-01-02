#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
collections模块基本介绍
我们都知道，Python拥有一些内置的数据类型，比如str, int, list, tuple, dict， set等，
collections模块在这些内置数据类型的基础上，提供了几个额外的数据类型：
    1.namedtuple(): 生成可以使用名字来访问元素内容的tuple子类
    2.deque: 双端队列，可以快速的从另外一侧追加和推出对象
    3.Counter: 计数器，主要用来计数
    4.OrderedDict: 有序字典
    5.defaultdict: 带有默认值的字典
 

"""

from collections import Counter


def part_1():
    dics = Counter(['b', 'a', 'd', 'a'])  # 会自动排序
    print dics
    print Counter('aea fww ah eiu fhiuh wae ihrie ahiuh eiur ef'.split())


if __name__ == '__main__':
    part_1()