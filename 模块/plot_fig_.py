#!/usr/bin/python
# coding=utf-8
"""
直接使用Artists创建图表的标准流程如下：
    创建Figure对象
    用Figure对象创建一个或者多个Axes或者Subplot对象
    调用Axies等对象的方法创建各种简单类型的Artists

plot: 可用plt.plot?查看
    By default, each line is assigned a different style,默认会显示不同的颜色

axes：子图 Axes对象表示一个绘图区域，可以理解为子图。
axis: 轴

subplot(numRows, numCols, plotNum)
    subplot将整个绘图区域等分为numRows行* numCols列个子区域，
    这三个数都小于10的话，可以把它们缩写为一个整数，例如subplot(323)和subplot(3,2,3)是相同的

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# 直线
def img_1():
    plt.plot([-4, 4], [2.2, -0.5], c='r', linewidth=3.0)  # 画直线 两点确定一条直线
    plt.show()


def img_1_1():
    plt.axis([-1, 15, -2, 2])  # 是指定xy坐标的起始范围，它的参数是列表[xmin, xmax, ymin, ymax]
    x = np.arange(0, 10, 0.01)
    y = np.sin(x)
    z = np.cos(x ** 2)
    plt.plot(x, y, color='r', linestyle='-', label="sin(x)", linewidth=1)  # 可用plt.plot?查看
    plt.plot(x, z, color='b', linestyle='-', marker='|', label="cos(x^2)", linewidth=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("first figure")
    plt.legend()  # 显示图示
    plt.show()


def img_1_2():
    fg1 = plt.figure(1)  # 创建图表1  matplotlib的图像都位于figure中
    ax0 = fg1.add_subplot(111)
    fg2 = plt.figure(2)  # 创建图表2
    ax1 = fg2.add_subplot(211)  # 在图表2中创建子图1
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2 = fg2.add_subplot(212)  # 在图表2中创建子图2
    x = np.linspace(0, 3, 100)  # 100是元素个数 arange中的是步长
    for i in range(6):
        ax0.plot(x, np.exp(i * x) / 3)
        ax1.plot(x, np.sin(i * x))
        ax2.plot(x, np.cos(i * x))
    plt.show()


# 散点图
def img_2():

    # ML in 10 minutes plot
    # cheat to get the same "random" numbers
    np.random.seed(seed=99)

    # make some data up
    mean1 = [1, 2]
    mean2 = [-1, -1]
    cov1 = [[1.0, 0.0], [0.0, 0.5]]
    cov2 = [[1.0, 0.0], [0.0, 1.0]]

    # create some points
    x1 = np.random.multivariate_normal(mean1, cov1, 500)
    x2 = np.random.multivariate_normal(mean2, cov2, 500)

    plt.scatter(x1[:, 0], x1[:, 1], c='r', s=50)
    plt.scatter(x2[:, 0], x2[:, 1], c='b', s=50)
    plt.plot([-4, 4], [2.2, -0.5], c='g', linewidth=3.0)  # 画直线 两点确定一条直线

    plt.title("ML in One Picture")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")

    plt.show()


# 三维立体图
def img_3():
    x, y = np.mgrid[-2:2:20j, -2:2:20j]
    z = x ** 2 + y ** 2 - 2
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('rainbow'))
    ax.contour(x, y, z, zdim='z', offset=-2, cmap='rainbow')
    plt.show()


def img_3_1():
    # 定义figure
    fig = plt.figure()
    # 将figure变为3d
    ax = Axes3D(fig)

    # 定义x, y
    x = np.arange(-4, 4, 0.25)
    y = np.arange(-4, 4, 0.25)

    # 生成网格数据
    X, Y = np.meshgrid(x, y)

    # 计算每个点对的长度
    R = np.sqrt(X ** 2 + Y ** 2)
    # 计算Z轴的高度
    Z = np.sin(R)

    # 绘制3D曲面
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # 绘制从3D曲面到底部的投影
    # ax.contour(X, Y, Z, zdim='z', offset=-2, cmap="rainbow")
    ax.contour(X, Y, Z, zdim='z', offset=-2, cmap="rainbow")

    # 设置z轴的维度
    ax.set_zlim(-2, 2)

    plt.show()


def img_3_2():
    def randrange(n, vmin, vmax):
        """
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        """
        return (vmax - vmin) * np.random.rand(n) + vmin

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

if __name__ == '__main__':
    img_1_2()


