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

from sklearn.linear_model import LinearRegression
regressor_1 = LinearRegression()
regressor_1.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
X_p = poly_regressor.fit_transform(X)

regressor_2 = LinearRegression()
regressor_2.fit(X_p,Y)

plt.scatter(X,Y,color="blue")
plt.plot(X,regressor_1.predict(X),color="red")
plt.title("Finding Salary (Linear Regression)")
plt.xlabel("Designation")
plt.ylabel("Salary")
plt.show()



plt.scatter(X,Y,color="blue")
plt.plot(X,regressor_2.predict(poly_regressor.fit_transform(X)),color="red")
plt.title("Finding Salary (Polynomial Regression)")
plt.xlabel("Designation")
plt.ylabel("Salary")
plt.show()


regressor_1.predict([[6.5]])

regressor_2.predict(poly_regressor.fit_transform([[6.5]]))
