# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:32:48 2019

@author: Rajorshee
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_excel("Decision Tree Classification.xlsx")
X = dataset.iloc[:, 2:4].values
Y = dataset.iloc[:,4].values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.25,random_state = 65)



## Classifier Creation
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 62)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)


from matplotlib.colors import ListedColormap
X_set,Y_set = X_train,Y_train
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop = X_set[:,0].max() + 1,step = 0.01),
                    np.arange(start=X_set[:,1].min() -1, stop = X_set[:,1].max() + 1,step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0],X_set[Y_set == j,1],
                c = ListedColormap(('red','green'))(i),label=j)
plt.title('Decision Tree Classifier(Training Set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
X_set,Y_set = X_test,Y_test
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop = X_set[:,0].max() + 1,step = 0.01),
                    np.arange(start=X_set[:,1].min() -1, stop = X_set[:,1].max() + 1,step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0],X_set[Y_set == j,1],
                c = ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression(Test Set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
