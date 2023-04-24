import pandas as pd 
import numpy as np

dataset = pd.read_csv('Iris.csv')
print(dataset.info())
print(dataset.skew())
print(dataset.kurt())

print("The max value of Sepal Length is", dataset['SepalLengthCm'].max(), "& the min value of Sepal Length is", dataset['SepalLengthCm'].min())
print("The max value of Sepal Width is", dataset['SepalWidthCm'].max(), "& the min value of Sepal Width is", dataset['SepalWidthCm'].min())
print("The max value of Petal Length is", dataset['PetalLengthCm'].max(), "& the min value of Petal Length is", dataset['PetalLengthCm'].min())
print("The max value of Petal Width is", dataset['PetalWidthCm'].max(), "& the min value of Petal Width is", dataset['PetalWidthCm'].min())

print(dataset.corr())