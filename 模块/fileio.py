#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# 文件操作 os模块 file模块
#


import os
import sys


result = ['didntLike', 'smallDoses', 'largeDoses']
dataSet = []
if os.path.exists('a.txt'):
    with open('a.txt') as f:
        for line in f:
            line = line.strip().split('\t')
            line[3] = str(result.index(line[3]) + 1)
            dataSet.append('\t'.join(line)+'\n')
        f.close()
else:
    print 'file not exist!'


fb = open("b.txt", 'w+')
fb.writelines(dataSet)
fb.close()



# ---------------------------------------------------------------------
def readFile(filename):
    f = file(filename)
    while True:
        line = f.readline()
        if len(line) == 0:  # EOF
            break
        print line,
    f.close()

if len(sys.argv) < 2:
    print 'No action specified!'
    sys.exit()  # 默认参0, 0 successful

if sys.argv[1].startswith('--'):
    option = sys.argv[1][2:]  # 切片
    if option == 'version':
        print 'version 12.0'
    elif option == 'help':
        print 'this is the command help'
    else:
        print 'unknown option'
    sys.exit()
else:
    for filename in sys.argv[1:]:  # 列表切片
        readFile(filename)


# -------------------------------------------------------------------


print os.listdir(os.getcwd())


# ----------- os模块 ----------------------------------------
'''
得到当前工作目录，即当前Python脚本工作的目录路径: os.getcwd()

返回指定目录下的所有文件和目录名:os.listdir()

函数用来删除一个文件:os.remove()

删除多个目录：os.removedirs（r“c：\python”）

检验给出的路径是否是一个文件：os.path.isfile()

检验给出的路径是否是一个目录：os.path.isdir()

判断是否是绝对路径：os.path.isabs()

检验给出的路径是否真地存:os.path.exists()

返回一个路径的目录名和文件名:os.path.split()     eg os.path.split('/home/swaroop/byte/code/poem.txt') 结果：('/home/swaroop/byte/code', 'poem.txt') 

分离扩展名：os.path.splitext()

获取路径名：os.path.dirname()

获取文件名：os.path.basename()

运行shell命令: os.system()

读取和设置环境变量:os.getenv() 与os.putenv()

给出当前平台使用的行终止符:os.linesep    Windows使用'\r\n'，Linux使用'\n'而Mac使用'\r'

指示你正在使用的平台：os.name       对于Windows，它是'nt'，而对于Linux/Unix用户，它是'posix'

重命名：os.rename（old， new）

创建多级目录：os.makedirs（r“c：\python\test”）

创建单个目录：os.mkdir（“test”）

获取文件属性：os.stat（file）

修改文件权限与时间戳：os.chmod（file）

终止当前进程：os.exit（）

获取文件大小：os.path.getsize（filename）
'''