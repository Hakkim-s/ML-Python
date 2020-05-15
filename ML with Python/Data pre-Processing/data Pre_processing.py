# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:13:14 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('Data Pre-Processing.xlsx')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values



'''from sklearn.preprocessing import Imputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])'''

from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencode_X = LabelEncoder()
onehotencoder = OneHotEncoder()
X[:,0] =labelencode_X.fit_transform(X[:,0])
X = onehotencoder.fit_transform(X).toarray()




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
ct = ColumnTransformer([("City", OneHotEncoder(), [0])],    remainder = 'passthrough')
X = ct.fit_transform(X).toarray()