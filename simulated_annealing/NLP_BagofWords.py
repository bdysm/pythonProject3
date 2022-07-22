# https://python-course.eu/machine-learning/text-classification-in-python.php

import re
import os


def dict_merge_sum(d1, d2):
    """ Two dicionaries d1 and d2 with numerical values and
    possibly disjoint keys are merged and the values are added if
    the exist in both values, otherwise the missing value is taken to
    be 0"""

    return {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}


d1 = dict(a=4, b=5, d=8)
d2 = dict(a=1, d=10, e=9)

print(dict_merge_sum(d1, d2))

#########################################################################

def reverse(string):
    if len(string)<=1: return string
    return reverse(string[1:])+string[0]

print(reverse("hello")) # olleh

##########################################################################

str(2)



