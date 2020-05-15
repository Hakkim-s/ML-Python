# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 04:33:49 2019

@author: Rajorshee
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_excel("Polynomial Regression.xlsx")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)


from sklearn.svm import SVR
regressor = SVR(kernel ='rbf')
regressor.fit(X,Y)




regressor.predict([[6.5]])


plt.scatter(X,Y,color="blue")
plt.plot(X,regressor.predict(X),color="red")
plt.title("Finding Salary (Linear Regression)")
plt.xlabel("Designation")
plt.ylabel("Salary")
plt.show()
