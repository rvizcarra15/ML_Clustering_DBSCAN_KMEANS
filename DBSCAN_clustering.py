# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:59:13 2023

@author: saga
"""
#https://www.dominodatalab.com/blog/topology-and-density-based-clustering
#Density-Based Spatial Clustering of Applications with Noise (DBSCAN) 
#is a base algorithm for density-based clustering. 
#It can discover clusters of different shapes and sizes 
#from a large amount of data, which is containing noise and outliers

#Compared to centroid-based clustering like k-means, 
#density-based clustering works by identifying “dense” clusters of points,
#allowing it to learn clusters of arbitrary shape and identify outliers in the data

#Unlike k-means, DBSCAN does not require the number of clusters as a parameter.
#Rather it infers the number of clusters based on the data

#DBSCAN requires ε and minPts parameters for clustering. 
#The minPts parameter is easy to set. 
#The minPts should be 4 for two-dimensional dataset. 
#For multidimensional dataset, minPts should be 2 * number of dimensions. 
#For example, if your dataset has 6 features, set minPts = 12. 
#Sometimes, domain expertise is also required to set a minPts parameter.



from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\RVC\Academico\PYTHON\Ejercicios\DBSCAN\Wholesalecustomersdatatext.txt')
#Drop non-continuous variables
data.drop(["Channel", "Region"], axis = 1, inplace = True)

#matriz correlación
Corr_matrix= data.corr()
round(Corr_matrix,2)
sns.heatmap(Corr_matrix)
plt.show()


#going to use only two of these attributes
data2 = data[["Grocery", "Milk"]]
data2 = data.to_numpy().astype("float32", copy = False)

#Because the values of the data are in the thousands, 
#we are going to normalize each attribute by scaling it to 0 mean and unit variance.

stscaler = StandardScaler().fit(data)
data2 = stscaler.transform(data)

#visualization
#sns.scatterplot(x="Grocery", y="Milk", data=data);
ax = sns.scatterplot(x="Grocery", y="Milk", data=data)
ax.set_title("Correlación")
ax.set_xlabel("Grocery");
plt.show()

#*******************************identify optimal ε parameter**************


from sklearn.neighbors import NearestNeighbors
# n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
nbrs = NearestNeighbors(n_neighbors = 5).fit(data2)
# Find the k-neighbors of a point
neigh_dist, neigh_ind = nbrs.kneighbors(data2)
# sort the neighbor distances (lengths to points) in ascending order
# axis = 0 represents sort along first axis i.e. sort along row
sort_neigh_dist = np.sort(neigh_dist, axis = 0)

import matplotlib.pyplot as plt
k_dist = sort_neigh_dist[:, 4]
plt.plot(k_dist)
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.show()

#*******Identifying the exact knee point could be difficult visually.
from kneed import KneeLocator
kneedle = KneeLocator(x = range(1, len(neigh_dist)+1), y = k_dist, S = 1.0, 
                      curve = "concave", direction = "increasing", online=True)

# get the estimate of knee point
print("Exact knee point: ",(kneedle.knee_y))
kneedle.plot_knee()
plt.show()

#**********************************DBSCAN*****************************************

dbsc = DBSCAN(eps = 2.20, min_samples = 4).fit(data2)
# get cluster labels
labels_dbscan = dbsc.labels_
# check unique clusters
set(labels_dbscan)
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1}
# -1 value represents noisy points could not assigned to any cluster
from collections import Counter
print(Counter(labels_dbscan))

plt.scatter(data2[:, 0], data2[:, 1], c=labels_dbscan, cmap='viridis')
plt.xlabel('Grocery')
plt.ylabel('Milk')
plt.title('DBSCAN Clustering Results')
plt.show()

#**********************************DBSCAN*****************************************

#We will construct a DBSCAN object that requires a minimum of 15 data points 
#in a neighborhood of radius 0.5 to be considered a core point.

#As a rule of thumb, MinPts= 2xD (dimensions) can be used, 
#but it may be necessary to choose larger values for very large data, 
#for noisy data, or for data that contains many duplicates.

dbsc = DBSCAN(eps = .5, min_samples = 4).fit(data2)
labels_dbscan2 = dbsc.labels_
core_samples = np.zeros_like(labels_dbscan2, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True

plt.scatter(data2[:, 0], data2[:, 1], c=labels_dbscan2, cmap='viridis')
plt.xlabel('Grocery')
plt.ylabel('Milk')
plt.title('DBSCAN Clustering Results')
plt.show()

#**************************************K-MEANS*************************************
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Run K-means clustering with k=3
kmeans = KMeans(n_clusters=2, random_state=0).fit(data2)

# Get the cluster labels and centroid positions
labels_kmeans = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points with their corresponding cluster labels
plt.scatter(data2[:, 0], data2[:, 1], c=labels_kmeans, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='r')
plt.xlabel('Grocery')
plt.ylabel('Milk')
plt.title('K-means Clustering Results with Centroids')
plt.show()



