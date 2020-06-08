import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 4].values

#dummies para categoricals

dummies = pd.get_dummies(dataset['State'], drop_first=True);
dummies = dummies.iloc[:, :].values

X = np.concatenate((X, dummies), axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)


#model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicciones iniciales
y_predict = regressor.predict(X_test)

#backward elimination 
import statsmodels.api as sm
X2 = np.concatenate((np.ones((50,1)).astype(int), X), axis=1) #agrego el 1 por el theta0, el intercept que me pide el OLS

Xopt = X2[:, [0,1,2,3,4,5]] #todo por ahora
ols_regressor = sm.OLS(endog=y, exog=Xopt).fit() #fit con todos los features. alpha = 5%
ols_regressor.summary() #x5 me da 99 de P value, lo saco.

Xopt = X2[:, [0,1,2,3,4]]
ols_regressor = sm.OLS(endog=y, exog=Xopt).fit()
ols_regressor.summary()

Xopt = X2[:, [0,1,2,3]]
ols_regressor = sm.OLS(endog=y, exog=Xopt).fit()
ols_regressor.summary()

Xopt = X2[:, [0,1,3]]
ols_regressor = sm.OLS(endog=y, exog=Xopt).fit()
ols_regressor.summary()

Xopt = X2[:, [0,1]]
ols_regressor = sm.OLS(endog=y, exog=Xopt).fit()
ols_regressor.summary()


#con esta desicion tomada (RnD) armo el model

X_train, X_test, y_train, y_test = train_test_split(Xopt,y, test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_predict2 = regressor.predict(X_test)