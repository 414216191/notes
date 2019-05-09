#coding:utf-8

from sklearn.datasets import load_iris
import numpy as np
import operator
data = load_iris()

dataSet = data['data']
labels = data['target']
def classify(newX,dataSet,labels,k):
    numSmaples = dataSet.shape[0]
    diffMat = np.tile(newX,[numSmaples,1])-dataSet
    sqDiff = diffMat**2
    sqDistance = sqDiff.sum(axis=1)
    dis = sqDistance**0.5
    #对值从小到大排列返回的是对应的索引排序
    sortDis = dis.argsort()
    # print(sortDis)
    classCount = {}
    for i in range(k):
        Label = labels[sortDis[i]]
        classCount[Label] = classCount.get(Label,0) + 1
    sortedClass = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClass[0][0]

print(classify([1,2,3,4],dataSet,labels,3))
print(classify([4,3,2,1],dataSet,labels,3))