#coding:utf-8

from numpy import *
import time
import matplotlib.pyplot as plt
#sigmoid函数
def sigmoid(X):
    return 1.0/(1+exp(-X))

#逻辑回归训练过程
def trainLR(train_x,train_y,opts):
    start_time = time.time()

    numSamples,numFeatures = shape(train_x)
    #学习步长
    alpha = opts['alpha']
    #最大迭代次数
    maxIter = opts['maxIter']
    #初始化权重
    weights = ones((numFeatures,1))
    #迭代方式
    iterWay = opts['iterWay']
    # print("iterWay:%s"%iterWay)

    for k in range(maxIter):
        #梯度下降
        if iterWay ==  'graRise':
            predictY = sigmoid(train_x*weights)
            error = train_y - predictY
            weights = weights + alpha * train_x.transpose() * error
        #随机梯度下降
        elif iterWay == 'stocGradRise':
            for i in range(numSamples):
                #计算梯度
                predictY = sigmoid(train_x[i,:]*weights)
                error =  train_y[i,0] - predictY
                weights = weights + alpha * train_x[i,:].transpose() * error
        elif iterWay == 'smoothStocGradRise':
            dataIndex = list(range(numSamples))
            for i in range(numSamples):
                alpha = 4.0/(1.0 + k + i) +0.01
                randIndex = int(random.uniform(0,len(dataIndex)))
                predictY = sigmoid(train_x[randIndex,:] * weights)
                error = train_y[randIndex,0] - predictY
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del(dataIndex[randIndex])
        else:
            raise NameError('Not support optimize method type')

    print("training set finished after %s seconds."%str(time.time()-start_time))
    return weights

def testLR(weights,test_x,test_y):
    numSamples,numFeatures = shape(test_x)
    matchCount = 0
    for i in range(numSamples):
        # print(test_x[i,:])
        predict = sigmoid(test_x[i,:] * weights) > 0.5
        #如果匹配则matchCount加一
        if predict == bool(test_y[i,0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy

def showLR(weights,train_x,train_y):
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 3:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    for i in range(numSamples):
        if int(train_y[i,0]) == 0:
            plt.plot(train_x[i,1],train_x[i,2],'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    min_x = min(train_x[:, 1])[0, 0]
    # print(min(train_x[:, 1]))
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


def loadData():
    train_x = []
    train_y = []
    fileIn = open('./testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0,float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()


## step 1: load data
print("step 1: load data...")
train_x, train_y = loadData()
test_x = train_x
test_y = train_y

## step 2: training...
print("step 2: training...")
opts = {'alpha': 0.01, 'maxIter': 20, 'iterWay': 'smoothStocGradRise'}
optimalWeights = trainLR(train_x, train_y, opts)
# print("weights:%s"%optimalWeights)
## step 3: testing
print("step 3: testing...")
accuracy = testLR(optimalWeights, test_x, test_y)

## step 4: show the result
print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
showLR(optimalWeights, train_x, train_y)