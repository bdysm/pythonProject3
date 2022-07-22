import nltk
import pandas as pd
import numpy as np

print(dir(nltk))

ps = nltk.PorterStemmer()

print(dir(ps))

a = [1, [2], 3]
b = a[1]
b.append(4)

print(a)

lst = [ ]
i = 0
i += 1
while i < 3:
    lst.insert(i, i) # [1, 2, 4].insert(2,2) # [1, 2, 2, 4]
print(lst)

#############################################################################
'''https://towardsdatascience.com/string-matching-with-fuzzywuzzy-e982c61f8a84

    https: // towardsdatascience.com / string - matching -
    with-fuzzywuzzy - e982c61f8a84'''

##############################################################################

#Installing FuzzyWuzzy
pip install fuzzywuzzy
#Import
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
Str_A = 'FuzzyWuzzy is a lifesaver!'
Str_B = 'fuzzy wuzzy is a LIFE SAVER.'
ratio = fuzz.ratio(Str_A.lower(), Str_B.lower())
print('Similarity score: {}'.format(ratio))

Str_A = 'Chicago, Illinois'
Str_B = 'Chicago'
ratio = fuzz.partial_ratio(Str_A.lower(), Str_B.lower())
print('Similarity score: {}'.format(ratio))

'''https://www.datasciencecentral.com/'''

https://www.geeksforgeeks.org/fuzzywuzzy-python-library/

https://github.com/seatgeek/thefuzz

https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_control_system_advanced.html

#example-plot-control-system-advanced-py





