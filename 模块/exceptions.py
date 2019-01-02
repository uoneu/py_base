#!/usr/bin/python
# coding=utf-8

#
# 异常处理
#

import sys

try:
    s = raw_input('Enter something --> ')
except EOFError:
    print '\nWhy did you do an EOF on me?'
    sys.exit()  # exit the program
except:
    print '\nSome error/exception occurred.'
    # here, we are not exiting the program

print 'Done'


# ----------------------------------------------------------------------


class ShortInputException(Exception):
    """A user-defined exception class."""
    def __init__(self, length, atleast):
        Exception.__init__(self)
        self.length = length
        self.atleast = atleast

try:
    s = raw_input('Enter something --> ')
    if len(s) < 3:
        raise ShortInputException(len(s), 3)  # 引发异常
    # Other work can continue as usual here
except EOFError:
    print '\nWhy did you do an EOF on me?'
except ShortInputException, x:
    print 'ShortInputException: The input was of length %d, \
          was expecting at least %d' % (x.length, x.atleast)
else:
    print 'No exception was raised.'


# --------------------------------------------------------------------


import time

try:
    f = open('a.txt')
    while True:
        line = f.readline()
        if len(line):
            break
        time.sleep(1)
        print line,
except:
    print 'some exception occured'
finally:
    f.close()
    print 'close the open file'
