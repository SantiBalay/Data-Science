#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:21:40 2020

@author: santiagobalay
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t', quoting=3) #ignoro ", uso tsv por las comas en oraciones

#limpio texto de palabras que no sirven, o que no me aportan info, como puntuaciones o 'the', 'on, 'was' etc.
#tambien aplico stemming, que me devuelve palabras en sus conjucaciones (?) iniciales. tipo loved, loving -> love . 
#el modelo que quiero armar se llama bag of words. voy a armar una matriz palabras por review, con 1 si esta presente y 0 sino.

import re



import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

ps = PorterStemmer()
#review = review.split()
#review = [word for word in review if (word not in stopwords.words('english'))]
#review = [ps.stem(word) for word in review]
#review = ' '.join(review)

corpus = [] #collecion de texto (reviews en este caso) del mismo tipo, lo voy a ir llenando.

for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  #parser que saca todo menos a-z A-Z y las reemplazo con espacio.
    review = review.lower()
    review = review.split()
    review = [word for word in review if (word not in stopwords.words('english'))]
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500 ) #podria haber usado parametros aca apra limpiar pero me di cuenta tarde. 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#si necesito scaling
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#                                           ------------------------------------------------
#naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#                                           ------------------------------------------------
#logistic regression - SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#                                           ------------------------------------------------
#SVM - SCALING
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#                                           ------------------------------------------------
#random forest - SCALING
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#                                           ------------------------------------------------

#testing
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
cm = confusion_matrix(y_test, y_pred)
accs= accuracy_score(y_test,y_pred)
ps=precision_score(y_test,y_pred)
rs=recall_score(y_test,y_pred)
print(f"Accuracy: {accs} | Precision: {ps} | Recall: {rs}")

#LOGISTIC REGRESSION : Accuracy: 0.75 | Precision: 0.7623762376237624 | Recall: 0.7475728155339806
#NAIVE BAYES : Accuracy: 0.73 | Precision: 0.6842105263157895 | Recall: 0.883495145631068
#SVM : Accuracy: 0.735 | Precision: 0.890625 | Recall: 0.5533980582524272
#RANDOM FOREST : Accuracy: 0.72 | Precision: 0.8507462686567164 | Recall: 0.5533980582524272
#--

