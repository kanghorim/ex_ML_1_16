# -*- coding: utf-8 -*-
"""exam04_pandas03.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q-wdNN-3bnnjebufceZG3tKn_3lGIGhf

apply()
"""

import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'fare']]
df['ten'] = 10
print(df.head())

def add_5(n):
    return n + 5

for i in range(len(df)):
    df.iloc[i, 0] = add_5(df.iloc[i, 0])
print(df.age)

sr1 = df['age'].apply(add_5)
print(sr1)

sr1 = df['age'].apply(lambda x: x * 3 - 4)
print(sr1)

df1 = pd.DataFrame({'a':['a0', 'a1', 'a2', 'a3'],
                    'b':['b0', 'b1', 'b2', 'b3'],
                    'c':['c0', 'c1', 'c2', 'c3']},
                   index=[0, 1, 2, 3])
print(df1)

df2 = pd.DataFrame({'a':['a2', 'a3', 'a4', 'a5'],
                    'b':['b2', 'b3', 'b4', 'b5'],
                    'c':['c2', 'c3', 'c4', 'c5'],
                    'd':['d2', 'd3', 'd4', 'd5']},
                   index=[2, 3, 4, 5])
print(df2)

result1 = pd.concat([df1, df2])
result1.reset_index(drop=True,  inplace=True)
print(result1)

result1 = pd.concat([df1, df2], ignore_index=True)
print(result1)

result2 = pd.concat([df1, df2], axis='columns')
print(result2)

result3 = pd.concat([df1, df2], axis='columns', join='inner')
print(result3)

result3 = pd.concat([df1, df2], axis='columns', join='outer')
print(result3)

df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]
print(len(df))
print(df.head())

print(df['class'].unique())
print(df['class'].value_counts())

grouped = df.groupby(['class'])
print(grouped)

grouped_df = {}
for key, group in grouped:
    print('key :', key)
    print('length :', len(group))
    grouped_df[key] = group
    print(group.head())

print(grouped_df.keys())

print(grouped_df['First'])

average = grouped.mean()
print(average)

average = grouped.min()
print(average)

average = grouped.max()
print(average)

grouped_two = df.groupby(['class', 'sex'])
for key, group in grouped_two:
    print('key : ', key)
    print('length :', len(group))
    print(group.head())

print(grouped_two.mean())

group3f = grouped_two.get_group(('Third', 'female'))
print(group3f.head())

dfg = grouped_two.mean()
print(dfg)

print(dfg.loc[('First', 'male')])

print(dfg.loc['First'])

print(dfg.xs('female', level='sex'))

df

pdf1 = pd.pivot_table(df, 
                      index='class',
                      columns='sex',
                      values='age',
                      aggfunc='mean')
print(pdf1)
print(type(pdf1))

pdf2 = pd.pivot_table(df, 
                      index='class',
                      columns='sex',
                      values=['age', 'fare'],
                      aggfunc='mean')
print(pdf2)
print(type(pdf2))

pdf3 = pd.pivot_table(df, 
                      index='class',
                      columns='sex',
                      values='age',
                      aggfunc=['mean', 'sum', 'std'])
print(pdf3)
print(type(pdf3))

pdf4 = pd.pivot_table(df, 
                      index=['class', 'sex'],
                      columns='survived',
                      values=['age', 'fare'],
                      aggfunc=['mean', 'max'])
print(pdf4)
print(type(pdf4))

print(pdf4.index)
print(pdf4.columns)

print(pdf4.xs('First'))