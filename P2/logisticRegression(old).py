# Author: Yuhang (10, Mar)
# compile environment: python3.8

# Libraries: 

from dataLoading import *

import numpy as np
from pprint import pprint
from sklearn import datasets, linear_model
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# Specify pipeline
LRpip = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', LogisticRegression(solver='liblinear')),
])

# k-fold validation before tuning 
kf = KFold(n_splits=5, shuffle=True)
curr_fold = 0
acc_list = []

for train_idx, test_idx in kf.split(twenty_train.data):

    LRpip.fit(twenty_train.data, twenty_train.target)

    predicted = LRpip.predict(twenty_test.data)

    acc = accuracy_score(twenty_test.target, predicted)
    
    acc_list.append(acc)

    curr_fold += 1

print("Accuracy before tuning:", np.average(acc_list))


# k-fold validation after tuning using random search
params = {
    "clf__penalty": ['l1', 'l2'],
    "clf__C": [1.0, 2.0, 3.0],
    "clf__max_iter": [1000, 2000, 3000]
    }

turned_lr_random = RandomizedSearchCV(LRpip, param_distributions=params, cv=5)

turned_lr_random.fit(twenty_train.data, twenty_train.target)
best_estimator = turned_lr_random.best_estimator_

print('Best Penalty(random search):', turned_lr_random.best_estimator_.get_params()['clf__penalty'])
print('Best C(random search):', turned_lr_random.best_estimator_.get_params()['clf__C'])
print('Best iteration(random search):', turned_lr_random.best_estimator_.get_params()['clf__max_iter'])

y_estimated = turned_lr_random.predict(twenty_test.data)
acc = np.mean(y_estimated == twenty_test.target)
print("Accuracy after tuning:{}".format(acc))
