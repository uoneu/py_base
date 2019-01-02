#!/usr/bin/python
# -*- coding: UTF-8 -*-


from collections import defaultdict

ab = {'Swaroop': 'swaroopch@byteofpython.info',
      'Larry': 'larry@wall.org',
      'Matsumoto': 'matz@ruby-lang.org',
      'Spammer': 'spammer@hotmail.com'
      }

print ab.keys()
print ab.values()

for name, address in ab.items():
    print 'Contact %s at %s' % (name, address)

if 'Guido' in ab:  # 或者 ab.has_key('Guido')
    print "\nGuido's address is %s" % ab['Guido']



#  -------------------------------------------------------------------------
print
words = ['apple', 'bat', 'atom', 'book', 'orange']
by_letter = {}
for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else:
        by_letter[letter].append(word)
print by_letter


# --------------------------------------------------------------------------
print
by_letter = {}
for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)  # 代替上面的if-else-语句块
print by_letter


# ------------------------------------------------------------------------
print
by_letter = defaultdict(list)  # 设置词典的默认值为列表
for word in words:
    by_letter[word[0]].append(word)

print by_letter


# -------------------------------------------------------------------------
# 可哈希性, 键值必须是不可变对象，列表不可作为键值!, 字典的值是可变的
print hash('sting')
