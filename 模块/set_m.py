#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# 集合
#

print set([1, 2, 3, 4, 3])

print {1, 2, 2, 3, 3}

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}

print a | b  # 或
print a ^ b  # 异或
print a - b  # 差
print a & b  # 与


a.issubset(b)


# 更多方法　查看ipython