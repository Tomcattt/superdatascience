#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:42:00 2020

@author: quentin
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Mall dataset with pandas
dataset = pd.read_csv('/Users/quentin/Documents/AI/UDEMY_ML/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []

'''
'k means++' is a special initialisation to avoid the random initialisation trap
'''

for i in range(1,11):
    KM = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    KM.fit(X)
    wcss.append(KM.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

'''
the good number of cluster is 5 according to the elbow method
'''

# Applying the k-means to the mall dataset
GoodKM = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = GoodKM.fit_predict(X)

# Plotting the clustured dataset (using boollean tricks)
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(GoodKM.cluster_centers_[:,0], GoodKM.cluster_centers_[:,1], s = 300, c = 'yellow' , label = 'Centroids')
plt.title('clusters of clients')
plt.ylabel('Spending Score')
plt.xlabel('Annual Income')
plt.legend()
plt.show()










