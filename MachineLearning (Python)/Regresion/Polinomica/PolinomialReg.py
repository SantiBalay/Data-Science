#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

#tengo demasiados pocos datos, no voy a divir en test/train porque encima la variacion es muy grande. Podrian agarrar 2 random para test pero no lo voy a hacer

#scaling no necesario por las librerias que voy a usar.

#regresion lineal
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) 


#regresion polinomica
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

Xpol = poly_reg.fit_transform(X) #me agrega la intercept tmb

pol_regressor = LinearRegression()
pol_regressor.fit(Xpol,y)


Xtest = X[3,:]
Xtest = np.array(Xtest, dtype=float)
Xtest[0] = 6.5
Xtest = Xtest.reshape(1, -1)

X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color='red')
plt.plot(X_grid, pol_regressor.predict(poly_reg.fit_transform(X_grid)), color='blue')

ypred = pol_regressor.predict(poly_reg.fit_transform(Xtest))