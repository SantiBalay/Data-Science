#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:45:42 2020

@author: santiagobalay
"""


#SVR

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

#svm no incluye feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X) 
y = sc_y.fit_transform(y)



from sklearn.svm import SVR
regressor = SVR(kernel='rbf') #gaussian

regressor.fit(X,y)

x_test = np.array([[6.5]])
x_test = sc_X.transform(x_test)

y_pred = regressor.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
