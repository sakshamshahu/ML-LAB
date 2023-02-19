#Logistic Regression on IRIS Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import pearsonr
from sklearn.inspection import DecisionBoundaryDisplay

sns.set(style="dark", color_codes=True)

dataset = pd.read_csv("Iris.csv")
dataset = dataset.drop(columns= "Id")
num_dataset =dataset.copy()
label = LabelEncoder()
num_dataset['Species'] = label.fit_transform(num_dataset['Species'])
print(num_dataset)

X = num_dataset.drop(columns = ['Species'])
Y = num_dataset['Species']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

model = LogisticRegression().fit(X_train,Y_train)
print(model.score(X_test,Y_test)*100,'%')

# sns.residplot(x='Species' , y='PetalLengthCm', data=num_dataset )
# plt.show()
g2 = sns.regplot(x='PetalLengthCm', y='Species', logistic=True,
   n_boot=750, y_jitter=.03, data=num_dataset,
   line_kws={'color': 'r'})
plt.show();