# Author: Yuhang (10, Mar)
# compile environment: python3.8

# Libraries: 

from dataLoading import twenty_train_data, twenty_train_target, twenty_test_data, twenty_test_target, IMDb_train_data, IMDb_train_target, IMDb_test_data, IMDb_test_target

import numpy as np
from pprint import pprint
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
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
('clf', LogisticRegression()),
])


kf = KFold(n_splits=5, shuffle=True)
curr_fold = 0
acc_list = []

for train_idx, test_idx in kf.split(twenty_train_data):

    LRpip.fit(twenty_train_data, twenty_train_target)

    predicted = LRpip.predict(twenty_test_data)

    acc = accuracy_score(twenty_test_target, predicted)
    
    acc_list.append(acc)

    curr_fold += 1

print("Accuracy before tuning:", np.average(acc_list))