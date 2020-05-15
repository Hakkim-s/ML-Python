# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 02:16:34 2019

@author: Rajorshee
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_excel("Polynomial Regression.xlsx")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values





regressor.predict([[6.5]])


plt.scatter(X,Y,color="blue")
plt.plot(X,regressor.predict(X),color="red")
plt.title("Finding Salary (Linear Regression)")
plt.xlabel("Designation")
plt.ylabel("Salary")
plt.show()


X_hr = np.arange(min(X),max(X),0.01)
X_hr = X_hr.reshape((len(X_hr),1))
plt.scatter(X,Y,color="blue")
plt.plot(X_hr,regressor.predict(X_hr),color="red")
plt.title("Finding Salary (Decision Tree)")
plt.xlabel("Designation")
plt.ylabel("Salary")
plt.show()
