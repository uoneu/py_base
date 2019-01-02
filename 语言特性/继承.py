#!/usr/bin/python
# coding=utf-8

#
# 面向对象的编程
# 继承
#


class Person:
    """docstring for Person"""

    population = 0  # 类变量 静态变量

    def __init__(self, name='none'):
        """Initializes the person's data.
        类似于Ｃ＋＋中的构造函数
        """
        self.name = name  # name是实例变量
        Person.population += 1
        print '(Initializing %s)' % self.name

    def __del__(self):  # 析构函数
        print 'I am dying %s ---' % self.name
        Person.population -= 1

        if Person.population == 0:
            print '  I am last one'
        else:
            print '  There are still %d people left.' % Person.population

    def sayHi(self):
        print 'hello, my name is %s' % self.name


p = Person('ycj')
p1 = Person('kimi')
del p1
p2 = Person('lucy')
p.sayHi()

print Person.population
print p.name  # 对象变量


# print Person.__init__.__doc__
# help(Person.__init__)
# print dir(Person)


# ---------------------------------------------------------------
print '\n'


class SchoolMember(object):
    """docstring for SchoolMember"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def tell(self):
        print 'Name: %s, Age: %d' % (self.name, self.age)


class Student(SchoolMember):
    """docstring for Student"""

    def __init__(self, name, age, marks):
        super(Student, self).__init__(name, age)
        self.marks = marks
        print 'student name is %s' % self.name

    def tell(self):
        SchoolMember.tell(self)
        print 'marks : "%d"' % self.marks


class Teacher(SchoolMember):
    """docstring for Teacher"""

    def __init__(self, name, age, salary):
        super(Teacher, self).__init__(name, age)
        self.salary = salary
        print 'teacher name is %s' % self.name

    def tell(self):
        super(Teacher, self).tell()  # == SchoolMember.tell(self)
        print 'salary : %d' % self.salary


stu = Student('ycl', 12, 100)
tc = Teacher('wang', 13, 500)

members = [stu, tc]
for member in members:
    member.tell()  # 多态

print '\n\n\n'
