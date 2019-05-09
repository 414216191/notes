#coding:utf-8

from sklearn.datasets import load_iris
data = load_iris()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
clf = AdaBoostClassifier(n_estimators=100)
score = cross_val_score(clf,data.data,data.target)
print(score.mean())