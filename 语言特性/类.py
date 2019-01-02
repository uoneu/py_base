#!/usr/bin/python
# coding=utf-8


#
# 访问属性
#

class Point(object):
    nums = 0
    __slots__ = ('__x', '__y', '_area')  # 用tuple定义允许绑定的属性名称, 限制实例对象绑定属性

    def __init__(self, x, y):
        self.__x, self.__y = x, y
        self._area = x * y
        Point.nums += 1

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = 0 if x<0 else x

    def __str__(self):
        """  打印实例  """
        return "Point : " + str([self.__x, self.__y, Point.nums])   # 可以直接打印实例变量

    __repr__ = __str__


def t_1():
    p = Point(1, 2)
    # 变量的绝对私有不可能, 双下划线可保证实例变量不能直接访问
    # 单下划线是一种约定，外部不要直接访问的信号！
    p._Point__x = 2
    p._area
    p1 = Point(2, 4)
    # 对象p绑定了新的变量，可用id(p.nums)查看，屏蔽了原类变量(原是所有对象共享，id()都一样)， p.ccc同样正确，新变量的绑定
    p.nums = 9
    p.tell()
    print p.nums


def t_2():
    """ 类似get/set, 不让属性暴露，可检查参数，防止直接修改属性 """
    p = Point(2, 5)
    print p.x
    p.x = 99
    print p.x
    p


if __name__ == '__main__':
    t_2()
