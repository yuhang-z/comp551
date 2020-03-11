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
from sklearn.tree import DecisionTreeClassifier


# Specify pipeline
DTpip = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', DecisionTreeClassifier()),
])



# k-fold validation before tuning 

kf = KFold(n_splits=5, shuffle=True)
curr_fold = 0
acc_list = []

for train_idx, test_idx in kf.split(twenty_train.data):

    DTpip.fit(twenty_train.data, twenty_train.target)

    predicted = DTpip.predict(twenty_test.data)

    acc = accuracy_score(twenty_test.target, predicted)
    
    acc_list.append(acc)

    curr_fold += 1

print("Accuracy before tuning:", np.average(acc_list))



# k-fold validation after tuning using random search
params = {
    "clf__max_depth": [400],
    "clf__max_features": [None, 'auto', 'log2'],
    "clf__criterion": ["gini", "entropy"],
    "clf__min_samples_split": [14, 15, 16, 17, 18, 19, 20, 21, 22],
    "clf__min_samples_leaf": [12, 14, 16, 17, 18, 19, 20, 22, 24],
    "clf__class_weight": ["balanced", None]
    }

turned_dt_random = RandomizedSearchCV(DTpip, param_distributions=params, cv=5)

turned_dt_random.fit(twenty_train.data, twenty_train.target)
best_estimator = turned_dt_random.best_estimator_

print('Best max_features(random search):', turned_dt_random.best_estimator_.get_params()['clf__max_features'])
print('Best criterion(random search):', turned_dt_random.best_estimator_.get_params()['clf__criterion'])
print('Best min_samples_split(random search):', turned_dt_random.best_estimator_.get_params()['clf__min_samples_split'])
print('Best min_samples_leaf(random search):', turned_dt_random.best_estimator_.get_params()['clf__min_samples_leaf'])
print('Best class_weight(random search):', turned_dt_random.best_estimator_.get_params()['clf__class_weight'])

y_estimated = turned_dt_random.predict(twenty_test.data)
acc = np.mean(y_estimated == twenty_test.target)
print("Accuracy after tuning:{}".format(acc))