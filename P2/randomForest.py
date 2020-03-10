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
from sklearn.ensemble import RandomForestClassifier


# Specify pipeline
RFpip = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', RandomForestClassifier(random_state=0)),
])



# k-fold validation before tuning 


kfold = KFold(n_splits=5, shuffle=False)

results = cross_val_score(RFpip, twenty_train.data, twenty_train.target, cv=kfold)

print("Accuracy before tuning:", results.mean())



# k-fold validation after tuning using random search
params = {
    "clf__bootstrap": [True, False],
    "clf__max_features": ['auto', 'sqrt'],
    "clf__n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    "clf__min_samples_split": [12, 14, 16, 18, 20],
    "clf__min_samples_leaf": [12, 14, 16, 18, 20],
    }

turned_rf_random = RandomizedSearchCV(RFpip, param_distributions=params, cv=5)

turned_rf_random.fit(twenty_train.data, twenty_train.target)
best_estimator = turned_rf_random.best_estimator_

print('Best bootstrap(random search):', turned_rf_random.best_estimator_.get_params()['clf__bootstrap'])
print('Best max_features(random search):', turned_rf_random.best_estimator_.get_params()['clf__max_features'])
print('Best n_estimators(random search):', turned_rf_random.best_estimator_.get_params()['clf__n_estimators'])
print('Best min_samples_split(random search):', turned_rf_random.best_estimator_.get_params()['clf__min_samples_split'])
print('Best min_samples_leaf(random search):', turned_rf_random.best_estimator_.get_params()['clf__min_samples_leaf'])

y_estimated = turned_rf_random.predict(twenty_test.data)
acc = np.mean(y_estimated == twenty_test.target)
print("Accuracy after tuning:{}".format(acc))