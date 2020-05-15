# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 06:14:05 2019

@author: Rajorshee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_excel("Random Forest Regression.xlsx")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10,random_state = 62 )
regressor.fit(X,Y)



Y_pred = regressor.predict([[6.5]])




X_hr = np.arange(min(X),max(X),0.01)
X_hr = X_hr.reshape((len(X_hr),1))
plt.scatter(X,Y,color="blue")
plt.plot(X_hr,regressor.predict(X_hr),color="red")
plt.title("Finding Salary (Decision Tree)")
plt.xlabel("Designation")
plt.ylabel("Salary")
plt.show()
