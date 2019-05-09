#coding:utf-8
import pandas as pd
filename = r'./sales_data.xls'
data = pd.read_excel(filename,index_col= r'序号')

# from sklearn.datasets import load_iris
# data = load_iris()
# print(data)
# data = pd.DataFrame(data)

#离散并量化
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data !=1 ] = -1
print(data)
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)
print(type(x))

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')
dtc.fit(x,y)

#可视化
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
with open("tree.dot",'w') as f:
    f = export_graphviz(dtc,feature_names=data.columns,out_file=f)
