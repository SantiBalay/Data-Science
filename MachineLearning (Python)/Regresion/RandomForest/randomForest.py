#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
#dataset
dataset = pd.read_csv('Position_Salaries.csv')

#dummies para categoricals

#dummies = pd.get_dummies(dataset['Country']);

#dummies = dummies.iloc[:, :].values
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10000, criterion='mse', random_state = 0)

regressor.fit(X,y)


regressor. 

x_pred = np.array([[6.5]])

y_pred = regressor.predict(x_pred)


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

