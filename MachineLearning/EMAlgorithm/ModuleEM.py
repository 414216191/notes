#coding:utf-8
"""
Author:Fang Long
Date:2017-12-22
"""
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
# heights_g = np.random.randn(1,50)
# heights_b = np.random.randn(1,50)
# print(heights_b)
# print(heights_g)

print(__doc__)

iris = load_iris()
skf = StratifiedKFold(n_splits=4)
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))
X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))
estimators = [GaussianMixture(n_components=n_classes,covariance_type=cov_type, max_iter=20, random_state=0) for cov_type in ['spherical', 'diag', 'tied', 'full']]
for estimator in estimators:
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                    for i in range(n_classes)])
    # print(estimator.means_init)
    estimator.fit(X_train)
    y_train_pred = estimator.predict(X_train)
    print("predict:%s"%estimator.predict_proba(X_train))
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print("train_accuracy:%s"%train_accuracy)
    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print("test_accuracy:%s" % test_accuracy)
