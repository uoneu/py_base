#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# 函数式编程
# 匿名函数　lambda
# map filter reduce
# 三元运算符
# 

add_ = lambda x, y: x + y  # 匿名绑定

print add_(4, 9)


# map filter reduce
def mrf():
    myls = [1, 2, 3]
    print map(lambda x: x + 1, myls)
    print filter(lambda x: x > 1, myls)
    print reduce(lambda x, y: x + y, myls, 10)


# 对象持久化
def persist_obj():
    import pickle as p  # 以便于我们可以使用更短的模块名称
    shoplistfile = 'shoplist.data'
    shoplist = ['adf', 'afaf', 'afed']
    f = file(shoplistfile, 'w')
    p.dump(shoplist, f)  # 转储
    f.close()

    del shoplist

    f = open(shoplistfile)
    shoplist = p.load(f)
    f.close()
    print shoplist