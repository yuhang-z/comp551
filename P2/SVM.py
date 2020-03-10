# Author: Yuhang (10, Mar)
# compile environment: python3.8

# Libraries: 

from dataLoading import *

import numpy as np
from pprint import pprint
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


# Specify pipeline
SVMpip = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', LinearSVC(random_state=0)),
])



# k-fold validation before tuning 


kfold = KFold(n_splits=5, shuffle=False)

results = cross_val_score(SVMpip, twenty_train.data, twenty_train.target, cv=kfold)

print("Accuracy before tuning:", results.mean())



# k-fold validation after tuning using random search
params = {
    "clf__C": [0.1, 1, 2, 3, 10, 100, 1000],
    "clf__loss": ['hinge', 'squared_hinge'],
    "clf__tol": [1e-3, 1e-4, 1e-5, 1e-6],
    "clf__max_iter": [1000, 2000, 3000],
    "clf__fit_intercept": [True, False],
    "clf__multi_class": ['ovr', 'crammer_singer']
    }

turned_svm_random = RandomizedSearchCV(SVMpip, param_distributions=params, cv=5)

turned_svm_random.fit(twenty_train.data, twenty_train.target)
best_estimator = turned_svm_random.best_estimator_

print('Best C(random search):', turned_svm_random.best_estimator_.get_params()['clf__C'])
print('Best loss(random search):', turned_svm_random.best_estimator_.get_params()['clf__loss'])
print('Best tol(random search):', turned_svm_random.best_estimator_.get_params()['clf__tol'])
print('Best max_iter(random search):', turned_svm_random.best_estimator_.get_params()['clf__max_iter'])
print('Best fit_intercept(random search):', turned_svm_random.best_estimator_.get_params()['clf__fit_intercept'])
print('Best multi_class(random search):', turned_svm_random.best_estimator_.get_params()['clf__multi_class'])

y_estimated = turned_svm_random.predict(twenty_test.data)
acc = np.mean(y_estimated == twenty_test.target)
print("Accuracy after tuning:{}".format(acc))