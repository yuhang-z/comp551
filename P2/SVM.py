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
('clf', LinearSVC(random_state=0, tol=1e-5)),
])



# k-fold validation before tuning 
# kf = KFold(n_splits=5, shuffle=True)
# curr_fold = 0
# acc_list = []

# for train_idx, test_idx in kf.split(twenty_train.data):

#     SVMpip.fit(twenty_train.data, twenty_train.target)

#     predicted = SVMpip.predict(twenty_test.data)

#     acc = accuracy_score(twenty_test.target, predicted)
    
#     acc_list.append(acc)

#     curr_fold += 1

# print("Accuracy before tuning:", np.average(acc_list))



# k-fold validation after tuning using random search
params = {
    "clf__C": [1, 2, 3],
    "clf__loss": ['hinge', 'squared_hinge'],
    "clf__max_iter": [1000, 2000, 3000],
    "clf__fit_intercept": [True, False],
    "clf__multi_class": ['ovr', 'crammer_singer']
    }

turned_svm_random = RandomizedSearchCV(SVMpip, param_distributions=params, cv=5, verbose=10, random_state=42, n_jobs=-1)

turned_svm_random.fit(twenty_train.data, twenty_train.target)
best_estimator = turned_svm_random.best_estimator_

print('Best C(random search):', turned_svm_random.best_estimator_.get_params()['clf__C'])
print('Best loss(random search):', turned_svm_random.best_estimator_.get_params()['clf__loss'])
print('Best max_iter(random search):', turned_svm_random.best_estimator_.get_params()['clf__max_iter'])
print('Best fit_intercept(random search):', turned_svm_random.best_estimator_.get_params()['clf__fit_intercept'])
print('Best multi_class(random search):', turned_svm_random.best_estimator_.get_params()['clf__multi_class'])

y_estimated = turned_svm_random.predict(twenty_test.data)
acc = np.mean(y_estimated == twenty_test.target)
print("Accuracy after tuning:{}".format(acc))