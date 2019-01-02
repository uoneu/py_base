#!/usr/bin/python
# -*- coding: UTF-8 -*-

import re

# 字符串
print "Hello World jisfe  wsfi"
print "jseij"  # 和单引号一样print

print '''jawoiej
aoeirfjhi' wfg' sf""
hahh'''  # 三引号支持一个多行的字符串

print 'my\' name is \
jfjoijo'

# 会自动按字面意义级连接字符串 或者使用+
print 'he is a ' 'girl!'

# 自然字符串 示某些不需要如转义符那样的特别处理的字符串
# 一定要使用自然字符串处理正则表达式
print r"Newlines are indicated by \n"


s1 = "  wer  we  wer  "
print s1.strip()  # j截掉开头和结尾处的空格
print s1.split()  # 默认分隔符为空字符（空格、换行、制表符）


listone = ['a', 'b', 'c']
print '-'.join(listone)
listtwo = 'a b v'
print listtwo.split(' ')
s = 'w e   e  e'
print "".join(s.split())
print re.sub(r'\s', '', s)