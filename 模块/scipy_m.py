#!/usr/bin/python
# coding=utf-8
"""
该模块包含概率论中的统计函数
"""

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np


def part_1():
    x = np.linspace(0, 5, 10, endpoint=False)
    y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(x, y)

    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, rv.pdf(pos), cmap='rainbow')
    plt.show()

if __name__ == '__main__':
    part_1()