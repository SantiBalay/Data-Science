#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:28:04 2020

@author: santiagobalay
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


#elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('Elbow')
plt.xlabel('Clusters')
plt.ylabel('WCSS')

#5 clusters parece ser lo correcto
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visualizo
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],s=100, c='red', label='Cluster1',edgecolors='black')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],s=100, c='blue', label='Cluster2',edgecolors='black')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],s=100, c='green', label='Cluster3',edgecolors='black')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],s=100, c='magenta', label='Cluster4',edgecolors='black')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1],s=100, c='cyan', label='Cluster5',edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids',edgecolors='black')
plt.title('Clusters of clients')
plt.xlabel('Anual income')
plt.ylabel('Spending Score')
plt.legend()