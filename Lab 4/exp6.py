#Multiple Linear Regression on boston housing
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import mean_squared_error 

boston = pd.read_csv('boston_housing.csv')
print(boston)
boston.hist(figsize=(16, 20),xlabelsize=12, ylabelsize=12)
plt.show()
sns.distplot(boston['medv'], color='g', bins=100, hist_kws={'alpha': 0.4});
plt.show()

#Finding the correlation with price
print(boston.corr()['medv'][:-1])

y=boston['medv']
fig, ax = plt.subplots(round(len(boston.columns) / 3), 3, figsize = (18, 12))
for i, ax in enumerate(fig.axes):
    if i < len(boston.columns) - 1:
        sns.regplot(x=boston.columns[i],y='medv',data=boston[boston.columns], ax=ax)
plt.show()

boston=boston.drop('medv',axis=1)
x_train,x_test,y_train,y_test = train_test_split(boston,y,test_size=0.3)

reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

plt.scatter(y_test, y_pred, c = 'blue') 
plt.show()

mse = mean_squared_error(y_test, y_pred) 
print("Mean Square Error : ", mse) 