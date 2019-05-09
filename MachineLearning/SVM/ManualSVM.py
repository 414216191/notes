#coding:utf-8

#读取鸢尾花数据
from sklearn.datasets import load_iris
import numpy as np
import random
import matplotlib.pyplot as plt

def kernel(x,xj,kernelOption):#高斯核函数（数据集线性不可分）
    #xi指的是整个数据集?
    #M指的是样本个数
    M = x.shape[0]
    K = np.zeros((M,1))
    kernelType =kernelOption[0]
    if kernelType=='linear':
        K = x * xj.T
    # A = xi - xj
    #高斯径向基函数
    if kernelType=='rbf':
        sigma = kernelOption[1]
        if sigma==0.0:
            sigma = 1.0
        for l in range(M):
            A = np.array(x[l])-xj
            K[l] = [np.exp(-0.5*float(A.dot(A.T))/(sigma**2))]
    else:
        raise NameError('Not support kernelType')
    return K

#记录每个样本与整体训练集的核函数结果
def kernelMatrix(x,kernelOption):
    numSamples = x.shape[0]
    Matrix = np.zeros((numSamples,numSamples))
    for i in range(numSamples):
        print("shape:%s"%str(Matrix[:,i].shape))
        Matrix[:,i] = kernel(x,x[i,:],kernelOption)
    return  Matrix

# print(kernel(positiveX,positiveX[0]))

#寻找在间隔边界上的支持向量
def find(alphas,C):
    supportVector = []
    for i in range(len(alphas)):
        if alphas[i] <C and alphas[i] >0:
            supportVector.append(alphas[i])
    return alphas[i]

#计算k对应的样本的预测值与真实值的差，用于选择内层循环
def error(svm,k):
    g_xk = float(np.multiply(svm.alphas,svm.Y)).T * svm.kernelMat[:,k] + svm.b
    y_k = float(svm.Y[k])
    return g_xk-y_k

#根据已选的alpha_i选择对应的alpha_j
def innerLoop(svm,i,error_i):
    #alpha_i的误差已经优化
    svm.errorCache[i] = [1,error_i]
    candidateAlphas = np.nonzero(svm.errorCache[:,0])[0]#保存更新状态为1的缓存项的行指标
    maxStep = 0
    alpha = 0
    error = 0
    if len(candidateAlphas)>1:
        for j in candidateAlphas:
            if j == i:
                continue
            error_j = error(svm,j)
            if (abs(error_i - error_j)) > maxStep:
                maxStep = abs(error_i - error_j)
                alpha = j
                error = error_j
    # if came in this loop first time, we select alpha j randomly
    else:
        alpha = i
        while i == alpha:
            alpha = int(random.uniform(0,svm.numSamples))
        error = error(svm,alpha)
    return alpha,error


#寻找违反kkt条件最大的那个点
def externalLoop(svm,i):
    error_i = error(svm,i)
    ### check and pick up the alpha who violates the KKT condition
    ## satisfy KKT condition
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
    ## violate KKT condition
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
    #违反kkt的情况
    if (svm.Y[i] * error_i < -svm.epsilon) and (svm.alphas[i]<svm.C) or (svm.Y[i] * error_i > svm.epsilon) and (svm.alphas[i] > 0):
        #找对应的alpha_j
        j,error_j = innerLoop(svm,i,error_i)
        alpha_i_old = svm.alphas[i].copy()
        alpha_j_old = svm.alphas[j].copy()
        #求解对应的L和H
        if svm.train_y[i] != svm.train_y[j]:
            L = max(0, svm.alphas[j] - svm.alphas[i])
            H = min(svm.C, svm.C + svm.alphas[j] - svm.alphas[i])
        else:
            L = max(0, svm.alphas[j] + svm.alphas[i] - svm.C)
            H = min(svm.C, svm.alphas[j] + svm.alphas[i])
        if L == H:
            return 0
        #求解最优解
        eta = svm.kernelMat[i,i] + svm.kernelMat[j,j] - 2 * svm.kernelMat[i,j]
        if eta  < 0:
            return 0
        svm.alpha[j] = alpha_j_old + float(svm.Y[j] * (error_i-error_j)/eta)
        if (svm.alpha[j] > H):
            svm.alpha[j] = H
        elif (svm.alpha[j] < L):
            svm.alpha[j] = L
        #更新错误率
        svm.errorCache[:,j] = [1,error(svm,j)]
        #若更新幅度太低，则直接返回
        if (abs(float(svm.alpha[j]) - alpha_j_old) < 0.00001): return 0
        #更新alpha_i
        svm.alphas[i] = alpha_i_old + svm.Y[i] * svm.Y[j] * (alpha_j_old  - svm.alpha[j])
        svm.errorCache[:, i] = [1, error(svm, i)]
        #更新b
        b1 = svm.b - error_i - svm.train_y[i] * (svm.alphas[i] - alpha_i_old)  * svm.kernelMat[i, i] - svm.train_y[j] * (svm.alphas[j] - alpha_j_old)  * svm.kernelMat[i, j]
        b2 = svm.b - error_j - svm.train_y[i] * (svm.alphas[i] - alpha_i_old) * svm.kernelMat[i, j]- svm.train_y[j] * (svm.alphas[j] - alpha_j_old)  * svm.kernelMat[j, j]
        if (0 < svm.alphas[i]) and (svm.alphas[i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[j]) and (svm.alphas[j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0
        return 1
    else:
        return  0

def SMO(train_x, train_y, C, epsilon, maxIter, kernelOption = ('rbf', 1.0)):
    svm = SVM(train_x,train_y,C,epsilon,kernelOption)
    iter = 0
    iterEntire = True  # 由于alpha被初始化为零向量，所以先遍历整个样本集
    while iter < maxIter:
        iter += 1
        if iterEntire:
            alphaChangePair = 0
            for i in range(svm.numSamples):
                alphaChangePair += externalLoop(svm,i)
            if alphaChangePair ==0:
                break
            else:
                iterEntire = False
        else:
            alphaChangePair = 0
            nonBound = find(svm.alphas,svm.C)
            for sv in nonBound:
                alphaChangePair += externalLoop(svm,sv)
            if alphaChangePair == 0:
                iterEntire = True
        return svm


def visualize(self, positive, negative):
    plt.xlabel('X1')  # 横坐标

    plt.ylabel('X2')  # 纵坐标

    plt.scatter(positive[:, 0], positive[:, 1], c='r', marker='o')  # +1样本红色标出

    plt.scatter(negative[:, 0], negative[:, 1], c='g', marker='o')  # -1样本绿色标出

    nonZeroAlpha = self.alphas[np.nonzero(self.alphas)]  # 非零的alpha

    supportVector = self.X[np.nonzero(self.alphas)[0]]  # 支持向量

    y = np.array([self.Y]).T[np.nonzero(self.alphas)]  # 支持向量对应的标签

    plt.scatter(supportVector[:, 0], supportVector[:, 1], s=100, c='y', alpha=0.5, marker='o')  # 标出支持向量

    print("支持向量个数:%d"%len(nonZeroAlpha))

    X1 = np.arange(-50.0, 50.0, 0.1)

    X2 = np.arange(-50.0, 50.0, 0.1)

    x1, x2 = np.meshgrid(X1, X2)

    g = self.b
    sigma = 10
    for i in range(len(nonZeroAlpha)):
        # g+=nonZeroAlpha[i]*y[i]*(x1*supportVector[i][0]+x2*supportVector[i][1])

        g += nonZeroAlpha[i] * y[i] * np.exp(
            -0.5 * ((x1 - supportVector[i][0]) ** 2 + (x2 - supportVector[i][1]) ** 2) / (sigma ** 2))

    plt.contour(x1, x2, g, 0, colors='b')  # 画出超平面

    plt.title("sigma: %f" % sigma)

    plt.show()

class SVM(object):
    def __init__(self,X,Y,C,epsilon,kernalOption):
        self.X = X
        self.Y = Y
        self.numSamples = X.shape[0]
        self.C = C
        self.epsilon = epsilon
        #初始alphas
        self.alphas = np.zeros((self.numSamples,1))
        self.b = 0
        self.kernelOption = kernalOption
        self.kernelMat = kernelMatrix(X,kernalOption)
        self.errorCache = np.zeros((self.numSamples,2))
if __name__ =="__main__":
    datas = load_iris()
    x = datas['data']
    y = datas['target']
    print((len(x)))
    positiveX = np.array([[0, 0, 0, 0]])  # y为0的样本集
    negativeX = np.array([[0, 0, 0, 0]])  # y为1和2的样本集
    for i in range(len(x)):
        if y[i] == 0:
            positiveX = np.row_stack((positiveX, np.array(x[i])))
        else:
            negativeX = np.row_stack((negativeX, np.array(x[i])))

    # print(len(positiveX[1:,:]))
    # print(len(negativeX[1:,:]))
    SVMClassifier=SMO(x,y,1,0.001,40)

    SVMClassifier.visualize(positiveX,negativeX)