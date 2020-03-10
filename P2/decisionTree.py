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

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier