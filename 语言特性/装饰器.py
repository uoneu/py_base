#!/usr/bin/python
# coding=utf-8
"""
    这种在代码运行期间动态增加功能的方式，称之为“装饰器” (Decorator)
    这种在代码运行期间动态增加功能的方式，称之为“装饰器” (Decorator)
    在面向对象（OOP）的设计模式中，decorator被称为装饰模式。
    OOP的装饰模式需要通过继承和组合来实现，而Python除了能支持OOP的decorator外，直接从语法层次支持decorator
    闭包
"""


def log(func):
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper


@log
def bar():
    print 'i am bar'


bar() # br=log(bar);br()
