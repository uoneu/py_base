#!/usr/bin/python
# -*- coding: UTF-8 -*-


#
# heapq模块   堆
#
# from…import *  把一个模块的所有内容全都导入到当前的命名空间
# from ...import name1, name2...
# import heapq as hq  # 以便于我们可以使用更短的模块名称
# 这不会直接把 heapq 中定义的函数的名字导入当前的符号表中；它只会把模块名字 heapq 导入其中。你可以通过模块名访问这些函数
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
x = heapq.heappop(heap)  # pops the smallest item from the heap
print heap[0]
