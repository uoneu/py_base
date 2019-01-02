#!/usr/bin/python
# -*- coding: UTF-8 -*-

#
# 正则表达
#
import re


# compile
# 返回一个正则对象, 常用的正则表达式编译成正则对象，可以提高效率
tt = "Tina is a good girl, she is cool, clever, and so on..."
rr = re.compile(r'\w*oo\w*')
print rr.findall(tt)


# match
# re.match(pattern, string, flags=0)  flag是匹配模式
line = "Cats are smarter than dogs"
mth = re.match(r'(?P<id>.*) are (.*?) .*', line, re.M | re.I)
print mth.group()
print mth.group(2)
print mth.group('id')


# search
# 在字符串内查找匹配，而match仅匹配开头
sc = re.search(r'dogs', line, re.M | re.I)
print sc.group()


# findall
# 返回一个列表
a = re.findall(r'\d+', '12 drumm44ers drumming, 11 ... 10 ...')
print a


# sub
# 检索和替换
# re.sub(pattern, repl, string, count=0, flags=0)
phone = "2004-959-559 # 这是一个国外电话号码"
num = re.sub(r'#.*$', "", phone)
print num
num = re.sub(r'\D', "", phone)
print num


# split
# 按照能够匹配的子串将string分割后返回列表
print(re.split('\d+', 'one1two2three3four4five5ww'))


# 匹配电话号码
p = re.compile(r'\d{3}-\d{6}')
print(p.findall('010-628888'))

# 匹配IP
re.search(r"(([01]?\d?\d|2[0-4]\d|25[0-5])\.){3}([01]?\d?\d|2[0-4]\d|25[0-5]\.)", "192.168.1.1")