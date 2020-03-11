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
('clf', DecisionTreeClassifier(random_state=0)),
])



# k-fold validation before tuning 


kfold = KFold(n_splits=5, shuffle=False)

results = cross_val_score(DTpip, twenty_train.data, twenty_train.target, cv=kfold)

print("Accuracy before tuning:", results.mean())



# k-fold validation after tuning using random search
params = {
    "clf__max_features": ['auto', 'sqrt', 'log2'],
    "clf__criterion": ["gini", "entropy"],
    "clf__min_samples_split": [2, 12, 14, 16, 18, 20],
    "clf__min_samples_leaf": [1, 12, 14, 16, 18, 20],
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