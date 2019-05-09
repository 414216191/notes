#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

def euclDistance(v1,v2):
    return sqrt(sum(power(v1-v2,2)))

def initCentroids(data,k):
    numSamples,numFeature = data.shape
    centroid = zeros((k,numFeature))
    for i in  range(k):
        ind = int(random.uniform(0,numSamples))
        centroid[i,:] = data[ind,:]
    # print(centroid)
    return centroid

def kMeans(data,k):
    numSmaples,numFeature = data.shape
    clusterAssessment = zeros((numSmaples,2))
    clusterChnage = True

    centroid = initCentroids(data,k)

    while clusterChnage:
        clusterChnage = False
        for i in range(numSmaples):
            minDis = 10000.0
            minIndex = 0
            for j  in range(k):
                dis = euclDistance(data[i,:],centroid[j,:])
                if dis < minDis:
                    minDis = dis
                    minIndex = j
            if clusterAssessment[i,0] != minIndex:
                clusterChnage = True
                clusterAssessment[i,:] = minIndex,minDis**2

        #更新中心点
        for j in range(k):
            pointsInCluster = data[nonzero(clusterAssessment[:, 0] == j)[0]]
            centroid[j,:] = mean(pointsInCluster,axis=0)

    print('Congratulations, cluster complete!')
    return centroid, clusterAssessment


def showCluster(dataSet, k, centroid, clusterAssessment):
    numSamples, dim = dataSet.shape

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssessment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroid[i, 0], centroid[i, 1], mark[i], markersize=12)

    plt.show()

# from sklearn.datasets import load_iris
# data = load_iris()
# dataSet = data['data']
# labels = data['target']
dataSet = random.rand(100,2)
centroid, clusterAssessment = kMeans(dataSet,3)
# print(centroid)
showCluster(dataSet,3,centroid,clusterAssessment)
