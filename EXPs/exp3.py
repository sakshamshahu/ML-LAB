import pandas as pd

dataset = pd.read_csv('datasets/Iris.csv')
print(dataset)

dt2 = pd.read_excel('datasets/Comments.xlsx')
print(dt2)

f = open('datasets/readlol.txt', 'r')
print(f.read())
f.close()