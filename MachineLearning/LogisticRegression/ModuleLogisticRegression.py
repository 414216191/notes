#coding:utf-8

import pandas as pd

filename = r'./bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()


print("x:%s"%x)
print("y:%s"%y)

#会报错，好像是个bug,因为并行化后数据块可能出现全为0的情况
#更改后，不会报错
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
#建立随机逻辑回归模型，筛选变量
rlr = RLR(sample_fraction=1)
rlr.fit(x,y)
print(rlr.get_support())
print("Feature Selection ended...")
print(data.iloc[:,0:8].columns)
print("Effective featrue is : %s"%(data.iloc[:,0:8].columns[rlr.get_support()]))

x = data[data.iloc[:,0:8].columns[rlr.get_support()]].as_matrix()#筛选好特征

lr = LR()
lr.fit(x,y)
print("LR training ended...")
print("Average correctness is :%s"%lr.score(x,y))

