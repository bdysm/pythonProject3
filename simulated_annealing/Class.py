import pandas as pd

s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])

"""
index
a       3
b       -5
c       7
d       4
"""
data = {'Country': ['Belgium', 'India', 'Brazil'],
        'Capital': ['Brussels', 'New Delhi', 'Brasília'],
        'Population': [11190846, 1303171035, 207847528]}

Index   Columns
        Country     Capital     Population
0
1
2

df = pd.DataFrame(data,
                columns=['Country', 'Capital', 'Population'])

help(pd.Series.loc)

s['b'] # Get one element

df[1:] # Get a subset of a DataFrame

# By Position
df.iloc[[0],[0]]

df.iat([0],[0])