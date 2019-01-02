#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# 列表 []
#
import numpy as np


ls_1 = [1, 2, 3, 4, 5, 6]
print ls_1[0]
print ls_1[0:-1]
print ls_1[1:-2]


print
# list 和 numpy数组的索引不同
ls_arr = np.array([[1, 2, 3], [4, 5, 6]])
print ls_arr
print ls_arr[0, 0]
print ls_arr[0, 0:2]


print
ls_2 = [1, 2, 3]
ls_2.reverse()  # 对象的方法都是对自己的数据操作，不是返回副本！ 对象的方法！！
print ls_2


print
print [1, 2, 3] * 2  # [1,2,3，1,2,3] 注意和numpy中的列表区分开
print [[]] * 3
print [1, 2, 3] + [2, 3, 4]


print
listone = [1, 2, 3, 4]
listtwo = [2 * i for i in listone if i >= 2]
print[elm for elm in listtwo if elm > 2]
print listtwo


# 切片
print [1, 2, 3, 4, 5][::2]  # 2表示步长
print [1, 2, 3, 4, 5][::-1]  # -表示方向 1表示步长，逆序操作