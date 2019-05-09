#coding:utf-8

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as  plt

data = np.random.rand(100,3)
print(data)
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
label = kmeans.labels_
print(label)
centroid = kmeans.cluster_centers_
print(centroid)
print(kmeans.inertia_)

numSamples = len(data)
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
for i in range(numSamples):
    #markIndex = int(clusterAssment[i, 0])
    plt.plot(data[i][0], data[i][1], mark[kmeans.labels_[i]]) #mark[markIndex])
mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
for i in range(3):
            plt.plot(centroid[i][0], centroid[i][1], mark[i], markersize = 12)
plt.show()