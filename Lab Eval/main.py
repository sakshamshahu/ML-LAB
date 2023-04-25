import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

data = pd.read_csv('updated.csv')

X= data.drop(['Comments', 'sentiments', 'class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))

y_true= y_test
cm = confusion_matrix(y_true, y_pred)

precision =precision_score(y_test, y_pred)
recall =  recall_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred)

print('Precision of Tree is : %.4f' %precision)
print('Recall of Tree is : %.4f' %recall)
print('F1 score of Tree is : %.4f' %f1)

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_lr")
plt.ylabel("y_true_lr")
plt.show()