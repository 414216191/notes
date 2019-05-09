#coding:utf-8
import numpy as np

def createDataSet():
    dataSet = [[1,1,0,'fight'],[1,0,1,'fight'],[1,0,1,'fight'],[1,0,1,'fight'],[0,0,1,'run'],[0,1,0,'fight'],[0,1,1,'run']]
    lables = ['weapon','bullet','blood']
    return dataSet,lables

#计算数据集的熵
def entropy(dataSet):
    numSamples = len(dataSet)
    labelCount = dict()
    for data in dataSet:
        label = data[-1]
        if label not in  labelCount.keys():
            labelCount[label] = 1.0
        else:
            labelCount[label]+=1.0
    entropyTemp = 0.0
    for numLabel in labelCount.values():
        pro = numLabel/numSamples
        entropyTemp -= pro * np.log2(pro)
    return entropyTemp

#给定特征，返回特征取某值的集合,feature代表特征所在的列的索引(离散特征）
#注意返回的数据集要去掉feature列
#离散变量则返回feature=value的行
def returnPartDataset(dataSet,feature,value):
    returnDataset = []
    for data in dataSet:
        if data[feature] == value:
            temp = data[:feature]
            temp.extend(data[feature+1:])
            returnDataset.append(temp)
    return returnDataset

#对于连续变量，则根据value指定的值进行划分
#连续变量则返回feature<value或者feature>value的行,direction指定,0返回大于value的数据集
def returnCountinousPartDataset(dataSet,feature,value,direction):
    returnDataset = []
    for data in dataSet:
        if direction ==0:
            if data[feature] > value:
                temp = data[:feature]
                temp.extend(data[feature+1:])
                returnDataset.append(temp)
        else:
            if data[feature] <= value:
                temp = data[:feature]
                temp.extend(data[feature+1:])
                returnDataset.append(temp)
    return returnDataset

#labels为标题行，此处用在修正特征时使用
def chooseBestFeature(dataSet,labels):
    print("dataSet:%s"%dataSet)
    #最后一列为label
    numFeatures = len(dataSet[0])-1
    Entropy_D = entropy(dataSet)
    info_gain = 0.0
    bestFeature = -1
    bestContinuousFeature = dict()
    for i in range(numFeatures):
        # 计算i的条件熵
        # 先看i的取值情况
        # 不能这么写
        # valueList  = set(dataSet[:,i])
        valueList = [exam[i] for exam in dataSet]
        #连续变量
        if type(valueList[0]).__name__=='float' or type(valueList[0]).__name__=='int':
            sortValueList = sorted(valueList)
            #产生n-1个划分点
            splitList = []
            #针对每一个划分值，计算其条件熵
            for j in sortValueList:
                ConditionEntropy = 0.0
                splitList.append((sortValueList[j]+sortValueList[j+1])/2.0)
                splitValue = (sortValueList[j]+sortValueList[j+1])/2.0
                smallDataset0 = returnCountinousPartDataset(dataSet,i,splitValue,0)
                pro0 =  len(smallDataset0)/float(len(dataSet))
                smallDataset1 = returnCountinousPartDataset(dataSet,i,splitValue,1)
                pro1 = len(smallDataset1) / float(len(dataSet))
                ConditionEntropy += pro0*entropy(smallDataset0) + pro1*entropy(smallDataset1)
                if info_gain < Entropy_D-ConditionEntropy:
                    info_gain = Entropy_D - ConditionEntropy
                    bestContinuousFeature[i] = splitValue
                    bestFeature = i
        #离散情况
        else:
            valueList = set([exam[i] for exam in dataSet])
            ConditionEntropy = 0
            for value in valueList:
                # 对每个value取数据集
                smallDataset = returnPartDataset(dataSet,i,value)
                pro = len(smallDataset)/float(len(dataSet))
                ConditionEntropy += pro * entropy(smallDataset)
            if info_gain<Entropy_D-ConditionEntropy:
                info_gain = Entropy_D-ConditionEntropy
                bestFeature = i

    #修正新特征
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestContinuousFeature[bestFeature]
        print("bestFeature:%d,bestSplitValue:%f"%(bestFeature,bestSplitValue))
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(len(dataSet)):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
        print("modifying dataSet:%s"%dataSet)
    return bestFeature

#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    return max(classCount)

#更新过，保证若训练数据集中有某个值划分后另一个特征有缺失的取值仍可以得到完整的决策树
def createTree(dataSet,labels,dataSetFull,labelsFull):
    #标签
    classList = [example[-1] for example in dataSet]
    #如果类标签中全为一个类，则返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果是最后一个属性，则返回
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeature(dataSet,labels)
    print("bestFeature choose:%d"%bestFeature)
    bestFeatureLabel = labels[bestFeature]
    Mytree = {bestFeatureLabel:{}}
    featureValues = [exam[bestFeature] for exam  in dataSet]
    featureSet = set(featureValues)
    if type(dataSet[0][bestFeature]).__name__=='str':
        currentlabel=labelsFull.index(labels[bestFeature])
        featValuesFull=[example[currentlabel] for example in dataSetFull]
        uniqueValsFull=set(featValuesFull)
    del(labels[bestFeature])
    print("labels:%s"%labels)
    for value in featureSet:
        if type(dataSet[0][bestFeature]).__name__=='str':
            uniqueValsFull.remove(value)
        subDataset  = returnPartDataset(dataSet,bestFeature,value)
        Mytree[bestFeatureLabel][value] = createTree(subDataset,labels,dataSetFull,labelsFull)
    if type(dataSet[0][bestFeature]).__name__=='str':
        #还有其他属性没有划分到，则使用当前数据集的投票机制决定空的属性取值的类
        for value in uniqueValsFull:
            Mytree[bestFeatureLabel][value]=majorityCnt(classList)
    return Mytree

data,labels = createDataSet()
print("data:%s"%data)
print("labels:%s"%labels)
# print(entropy(data))
# data[0][-1] = 'surrender'
# print(entropy(data))
print(createTree(data,labels,data,labels))