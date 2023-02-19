import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#import statsmodel as sm using which you can create variables x and y 
dataset = pd.read_csv('Summary of Weather.csv')

X = np.array(dataset['MinTemp']).reshape(-1,1)
print(X)
y = np.array(dataset['MaxTemp']).reshape(-1,1)

modelAlgo =  LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

modelAlgo.fit(X_train, y_train)

print('The model score is: ',(modelAlgo.score(X_test, y_test))*100, '%')
y_pred = modelAlgo.predict(X_test)
plt.scatter(X_test, y_test, color ='g', marker='*')
plt.plot(X_test, y_pred, color ='r')

plt.show()