#coding:utf-8
import numpy as np
def loadSimpData():
    dataMat=np.matrix([[1. ,2.1],
        [2. ,1.1],
        [1.3,1. ],
        [1. ,1. ],
        [2. ,1. ]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

#确定树桩分类器(dimen,thresholdValue,thresholdDire)确定，即对应的特征及特征值及特征方向，对数据进行分类
def weakClassify(data,dimen,thresholdValue,thresholdDire):
    #最开始所有的数据的类别都置为1
    Res = np.ones((np.shape(data)[0],1))
    #小于当前分裂值为-1类
    if thresholdDire=='lt':
        Res[data[:,dimen]<= thresholdValue]=-1.0
    else:
        Res[data[:, dimen] > thresholdValue] = -1.0
    return Res

#弱分类器的构造
def weakClassifier(data,labels,D):
    dataMatrix = np.matrix(data)
    labelMatrix = np.matrix(labels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestClassfier = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = 10000.0
    #遍历所有特征，确定一个最佳划分点
    for i in range(n):
        attr_min = dataMatrix[:,i].min()
        attr_max = dataMatrix[:,i].max()
        stepSize = (attr_max-attr_min)/numSteps
        #上面计算了当前属性的划分点
        for j in range(0,int(numSteps+1)):
            for dir in ['lt','gt']:
                thresValue = attr_min+float(j)*stepSize
                predictValue = weakClassify(dataMatrix,i,thresholdValue=thresValue,thresholdDire=dir)
                currentErr = np.mat(np.ones((m,1)))
                currentErr[predictValue==labelMatrix] = 0
                #最终加权以后的当前分类器的错误
                weightedErr = D.T*currentErr
                if weightedErr<minError:
                    minError = weightedErr
                    bestClasEst = predictValue.copy()
                    bestClassfier['dimen'] = i
                    bestClassfier['thresholdValue'] = thresValue
                    bestClassfier['thresDire'] = dir
    return bestClassfier,minError,bestClasEst

def adaboostTraining(data,classLabels,numIt = 40):
    weakClassifiers  = []
    m = np.shape(data)[0]
    #样本权重
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        classifier,error,classRes = weakClassifier(data,classLabels,D)
        #计算当前分类器的权重
        alpha = float(0.5*np.log((1-error)/max(error,1e-16)))
        classifier['alpha'] = alpha
        #更新样本权重
        expo = np.multiply(-1*alpha*np.mat(classLabels).T,classRes)
        D = np.multiply(D,np.exp(expo))
        D = D/D.sum()
        aggClassEst += alpha*classRes
        aggErrors = np.multiply(np.sign(aggClassEst)!= np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total Error:%s"%errorRate)
        weakClassifiers.append(classifier)
        if errorRate==0.0:
            break
    return weakClassifiers ,aggClassEst

data,classLabels = loadSimpData()
weakClassifiers ,aggClassEst = adaboostTraining(data,classLabels,5)
print("weakClassifiers:%s"%weakClassifiers)
print("aggClassEst:%s"%aggClassEst)
