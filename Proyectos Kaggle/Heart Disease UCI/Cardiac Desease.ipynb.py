# -*- coding: utf-8 -*-
"""Cardiac Desease.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k14QRsoAWMzJy-85mgIvOrUtMxYoChJb
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import statsmodels.regression.linear_model as sm
import seaborn as sns

!pip install -q kaggle
from google.colab import files

files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d ronitf/heart-disease-uci

!unzip heart-disease-uci.zip

!ls

df = pd.read_csv('heart.csv')

df.head()

df

"""# Columnas
1. Age: Edad de paciente
2. Sex: Sexo del paciente (1=Hombre, 0=Mujer)
3. Cp: Dolor en el pecho (1=,2=,3=)
4. Trestbps: Estudio en descanso de la presion de sangre 
5. Col: Colesterol en sangre
6. Fbs: Azucar en sangre en ayuno (1 = >120 mg/dl, 0 = no supera)
7. Restecg: Estudio en descanso cardiografico (1=positivo,0=negativo)
8. Thalanch: Maximo bps 
9. Exang: Angina inducida por ejercicio (1 = si, 0 = no)
10. Oldpeak: Depresion inducida en ejercicio con respecto a en descanso
11. Slope: Pendiente del maximo del BPS 
12. Ca: Numero de vasos sanguineos mayores coloreados en flouroscopia
13. Thal: Precencia de thalassemia (0=normal, 1=arreglado, 2=reversible)
14. Target: Existencia de problema cardiaco (1=Positivo, 0=Negativo)
"""

#Distribuciones
fig, ax = plt.subplots(3,5,figsize=(20,10))
plt.subplots_adjust(hspace = 0.8)
columnas = df.columns

sns.distplot(df[columnas[0]],ax=ax[0][0]);
ax[0][0].set_title('Edad', fontsize=14)
sns.countplot(df[columnas[1]],ax=ax[1][0]);
ax[1][0].set_title('Sexo',fontsize=14)
sns.countplot(df[columnas[2]],ax=ax[2][0]);
ax[2][0].set_title('Chest Pain',fontsize=14)
sns.distplot(df[columnas[3]],ax=ax[0][1]);
ax[0][1].set_title('Estudio en descanso',fontsize=14)
sns.distplot(df[columnas[4]],ax=ax[1][1]);
ax[1][1].set_title('Colesterol',fontsize=14)
sns.countplot(df[columnas[5]],ax=ax[2][1]);
ax[2][1].set_title('Azucar en sangre (Ayuno)',fontsize=14)
sns.countplot(df[columnas[6]],ax=ax[0][2]);
ax[0][2].set_title('Estudio Cardiografico',fontsize=14)
sns.distplot(df[columnas[7]],ax=ax[1][2]);
ax[1][2].set_title('Maximo Bps',fontsize=14)
sns.countplot(df[columnas[8]],ax=ax[2][2]);
ax[2][2].set_title('Angina inducida',fontsize=14)
sns.distplot(df[columnas[9]],ax=ax[0][3]);
ax[0][3].set_title('Oldpeak',fontsize=14)
sns.countplot(df[columnas[10]],ax=ax[1][3]);
ax[1][3].set_title('Pendiente del maximo BPS',fontsize=14)
sns.countplot(df[columnas[11]],ax=ax[2][3]);
ax[2][3].set_title('Numero de vasos Sanguineos',fontsize=14)
sns.countplot(df[columnas[12]],ax=ax[0][4]);
ax[0][4].set_title('Thalassemia',fontsize=14)
sns.countplot(df[columnas[13]],ax=ax[1][4]);
ax[1][4].set_title('Target',fontsize=14)
plt.show()

corr = df.corr()

fig, ax = plt.subplots(1,1,figsize=(20,10))

sns.heatmap(corr, cmap='coolwarm_r',annot_kws={'size':50})
ax.set_title('Heatmap de correlacion entre variables', fontsize=15)
plt.show()

"""#Preprocesamiento"""

sns.countplot('target', data=df,palette=['#0101DF','#DF0101']) #Balanceado, no aplico ningun metodo de balanceo
plt.title('Titulo')
plt.show()

df2 = df.copy()

#paso los categoricos a numericos (dropeando 1)

df2['dolor angina normal'] = df['cp'] == 1 
df2['dolor angina atipica'] = df['cp'] == 2 
df2['dolor no anginal'] = df['cp'] == 3 
df2 = df2.drop('cp', axis=1)
#4 es sin dolor -> 0 0 0
df2['rcg_abnormalidad_1'] = df['restecg'] == 1 
df2['rcg_abnormalidad_2'] = df['restecg'] == 2 
df2 = df2.drop('restecg',axis=1)
#tipo 0 es normal -> 0 0
df2['thal_defecto_corregido'] = df['thal'] == 2 
df2['thal_defecto_reversible'] = df['thal'] == 3 
df2 = df2.drop('thal',axis=1)
#tipo 1 es normal -> 0 0
df2['slope_subiendo'] = df['slope'] == 0
df2['slope_bajando'] = df['slope'] == 2
df2 = df2.drop('slope',axis=1)
#sin curva == 1 -> 0 0

df2 = df2*1 #booleans to ints

df2.head() #terminado

X = df2.iloc[:,0:19]
Xlen = X.shape[0]
Xones = np.ones(Xlen).reshape(-1,1)
X.insert(0, 'Const', Xones) #para el OLS
X.head()

#clases = X['target']
X = X.drop('target', axis=1)

from sklearn.preprocessing import RobustScaler
rc = RobustScaler()

X['age'] = rc.fit_transform(X['age'].values.reshape(-1,1))
X['trestbps'] = rc.fit_transform(X['trestbps'].values.reshape(-1,1))
X['chol'] = rc.fit_transform(X['chol'].values.reshape(-1,1))
X['thalach'] = rc.fit_transform(X['thalach'].values.reshape(-1,1))
X['oldpeak'] = rc.fit_transform(X['oldpeak'].values.reshape(-1,1))
X.head()

X = X.drop('Const', axis=1)

X

Xvals2 = X.values
print(Xvals2)
y = df['target'].values
y.size
Xvals2.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xvals2,y, test_size=0.2)

#Logistic Regresion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))
sns.heatmap(cm, annot=True)
plt.title('CM')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier2.fit(X_train,y_train)

y_pred2 = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test,y_pred2)

print(classification_report(y_test,y_pred2))
sns.heatmap(cm2, annot=True)
plt.title('CM')
plt.show()

from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators=10, criterion="entropy",random_state=0)
classifier3.fit(X_train,y_train)

y_pred3 = classifier3.predict(X_test)
cm3 = confusion_matrix(y_test,y_pred3)

print(classification_report(y_test,y_pred3))
sns.heatmap(cm3, annot=True)
plt.title('CM')
plt.show()

from sklearn.naive_bayes import GaussianNB
classifier4 = GaussianNB()
classifier4.fit(X_train,y_train)

y_pred4 = classifier4.predict(X_test)
cm4 = confusion_matrix(y_test,y_pred4)

print(classification_report(y_test,y_pred4))
sns.heatmap(cm4, annot=True)
plt.title('CM')
plt.show()

from sklearn.svm import SVC
classifier5 = SVC(kernel='rbf', random_state = 0, C=5)
classifier5.fit(X_train,y_train)

y_pred5 = classifier5.predict(X_test)
cm5 = confusion_matrix(y_test,y_pred5)

print(classification_report(y_test,y_pred5))
sns.heatmap(cm5, annot=True)
plt.title('CM')
plt.show()

!pip install pycaret
from pycaret.classification import *

X.head()

X['Target'] = y
cf = setup(X,target='Target')

compare_models() #consistente con lo hecho previamente