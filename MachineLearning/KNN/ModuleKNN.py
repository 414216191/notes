#coding:utf-8
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
data = load_iris()
x = data['data']
y = data['target']
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.4,random_state=1)
train_x1 = train_x[:,0]
train_x2 = train_x[:,1]
test_x1 = test_x[:,0]
test_x2 = test_x[:,1]

print("y:%s"%y)

# plt.figure(12)
plt.title("Iris")
plt.legend(['train','test'],loc='upper right')
plt.xlabel('x1')
plt.ylabel('x2')
plt.subplot(221)
plt.scatter(train_x1,train_x2,color='#9b59b6',marker='^',s=60)
plt.scatter(test_x1,test_x2,color ='#3498db',marker='o',s=60)
# plt.show()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x,train_y)

print("accuracy:%s"%knn.score(test_x,test_y))
print(knn.classes_)
y0 = []
y1 = []
y2 = []
for test in test_x:
    if knn.predict(test)==0:
        y0.append(test)
    elif knn.predict(test)==1:
        y1.append(test)
    elif knn.predict(test)==2:
        y2.append(test)
    else:
        raise NameError("No target!")

# print(np.mat(y0))
y0 = np.array(y0)
y1 = np.array(y1)
y2 = np.array(y2)
plt.subplot(222)
plt.scatter(y0[:,0],y0[:,1],color ='#3498db',marker='o',s=60)
plt.subplots(223)
plt.scatter(y1[:,0],y1[:,1],color ='#3498db',marker='o',s=60)
plt.subplot(224)
plt.scatter(y2[:,0],y2[:,1],color ='#3498db',marker='o',s=60)
plt.show()