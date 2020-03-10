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
from sklearn.ensemble import AdaBoostClassifier


# Specify pipeline
adapip = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', AdaBoostClassifier(random_state=0)),
])



# k-fold validation before tuning 


kfold = KFold(n_splits=5, shuffle=False)

results = cross_val_score(adapip, twenty_train.data, twenty_train.target, cv=kfold)

print("Accuracy before tuning:", results.mean())



# k-fold validation after tuning using random search
params = {
    "clf__n_estimators": [50, 100],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
    #"clf__loss": ['linear', 'square', 'exponential'],
    "clf__algorithm": ['SAMME', 'SAMME.R']
    }

turned_dt_random = RandomizedSearchCV(adapip, param_distributions=params, cv=5)

turned_dt_random.fit(twenty_train.data, twenty_train.target)
best_estimator = turned_dt_random.best_estimator_

print('Best n_estimators(random search):', turned_dt_random.best_estimator_.get_params()['clf__n_estimators'])
print('Best learning_rate(random search):', turned_dt_random.best_estimator_.get_params()['clf__learning_rate'])
#print('Best loss(random search):', turned_dt_random.best_estimator_.get_params()['clf__loss'])
print('Best algorithm(random search):', turned_dt_random.best_estimator_.get_params()['clf__algorithm'])

y_estimated = turned_dt_random.predict(twenty_test.data)
acc = np.mean(y_estimated == twenty_test.target)
print("Accuracy after tuning:{}".format(acc))