#coding:utf-8
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
datas = load_iris()
data = datas['data'][:,:2]
label = datas['target']
# print(data)
# print(label)

#划分测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(data,label,random_state=1,train_size=0.6)

#训练svm
'''
SVC参数解释 
（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0； 
（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF"; 
（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂； 
（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features; 
（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效； 
（6）probablity: 可能性估计是否使用(true or false)； 
（7）shrinking：是否进行启发式； 
（8）tol（default = 1e - 3）: svm结束标准的精度; 
（9）cache_size: 制定训练所需要的内存（以MB为单位）； 
（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应； 
（11）verbose: 跟多线程有关，不大明白啥意思具体； 
（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited; 
（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None 
（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。 
 ps：7,8,9一般不考虑。 
 '''
#注意这里是多分类，一共有3类，ovr 指的是one v rest，one v one
#注意是SVC
clf = svm.SVC(C=0.8,kernel='linear',gamma=20,decision_function_shape='ovr')
clf.fit(x_train,y_train)

def show_accuracy(pr,rr,strTest):
    count = 0
    for i in range(len(pr)):
        # print("pr:%d,rr:%d"%(pr[i],rr[i]))
        if pr[i]==rr[i]:
            count+=1
    print(strTest+r'的准确度为:%s'%str(count/len(pr)))

#查看分类效果
print(clf.score(x_train,y_train))
y_hat = clf.predict(x_train)
show_accuracy(y_hat,y_train,'训练集')
y_hat = clf.predict(x_test)
show_accuracy(y_hat,y_test,'测试集')

print('decision_function:\n', clf.decision_function(x_train))
print('\npredict:\n', clf.predict(x_train))