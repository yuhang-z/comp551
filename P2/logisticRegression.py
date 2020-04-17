# Author: Yuhang, Diyang
# compile environment: python3.8

# Libraries: 

from dataLoading import *

import numpy as np
from pprint import pprint
from sklearn import datasets, linear_model
from sklearn import metrics
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


### Specify pipeline
LRpip1 = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf',  LogisticRegression(solver='liblinear')),
])

LRpip2 = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf',  LogisticRegression(solver='liblinear')),
])


print("Default Parameters: solver='liblinear', which handles l1 Penalty")

### PART I (Possible Bonus): Perform Training on training set, Predictions also on training set
LRpip1.fit(twenty_train.data, twenty_train.target)
pred = LRpip1.predict(twenty_train.data)
print("(Bonus) 20: Training Set Accuracy:", metrics.f1_score(twenty_train.target, pred, average='macro'))

LRpip2.fit(imdb_train.data, imdb_train.target)
pred = LRpip2.predict(imdb_train.data)
print("(Bonus) imdb: Training Set Accuracy:", accuracy_score(imdb_train.target, pred))


### PART II (Required): Perform Training on training set, Predictions on test set
pred = LRpip1.predict(twenty_test.data)
print("(Required) 20: Test Set Accuracy:", metrics.f1_score(twenty_test.target, pred, average='macro'))

pred = LRpip2.predict(imdb_test.data)
print("(Required) imdb: Test Set Accuracy:", accuracy_score(imdb_test.target, pred))


### Part III (Required): K-Fold cross validation
print("(Required) 20: K-cv score before tuning:", cross_val_score(LRpip1, twenty_train.data, twenty_train.target, cv=5, scoring='accuracy').mean())

print("(Required) imdb: K-cv score before tuning:", cross_val_score(LRpip2, imdb_train.data, imdb_train.target, cv=5, scoring='accuracy').mean())
# kf = KFold(n_splits=5, random_state=None, shuffle=False)
# print(twenty_train.data.shape)
# for train_index, test_index in kf.split(twenty_train):
#   LRpip.fit(twenty_train.data[train_index], twenty_train.target[test_index])
#   pred = LRpip.predict(twenty_train.data[test_index])
#   print(metrics.f1_score(twenty_train.target[test_index], pred, average='macro'))

### Part IV (Bonus): A study of comparison between using different "super-class" of categories for training
### For 20newsgroup only, we compare the result of training using "comp.", "sci.", "rec." & "talk."
### Results are based on accuracies of predictions on test-set
comp = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x']
rec = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
sci = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
talk = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
comp_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=comp)
rec_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=rec)
sci_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=sci)
talk_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=talk)
comp_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=comp)
rec_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=rec)
sci_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=sci)
talk_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, categories=talk)
LRpip1.fit(comp_train.data, comp_train.target)
pred = LRpip1.predict(comp_test.data)
print("(Bonus) Comp. Test Set Accuracy:", metrics.f1_score(comp_test.target, pred, average='macro'))
LRpip1.fit(rec_train.data, rec_train.target)
pred = LRpip1.predict(rec_test.data)
print("(Bonus) Rec. Test Set Accuracy:", metrics.f1_score(rec_test.target, pred, average='macro'))
LRpip1.fit(sci_train.data, sci_train.target)
pred = LRpip1.predict(sci_test.data)
print("(Bonus) Sci. Test Set Accuracy:", metrics.f1_score(sci_test.target, pred, average='macro'))
LRpip1.fit(talk_train.data, talk_train.target)
pred = LRpip1.predict(talk_test.data)
print("(Bonus) Talk. Test Set Accuracy:", metrics.f1_score(talk_test.target, pred, average='macro'))


### Part V: Tuning Hyperparameters
# k-fold validation after tuning using random search
params = {
    "clf__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    }

gsearch1 = GridSearchCV(LRpip1, param_grid=params, cv=5)
gsearch1.fit(twenty_train.data, twenty_train.target)
print(gsearch1.best_params_)
print(gsearch1.best_score_)
pred = gsearch1.predict(twenty_test.data)
print("(Required) 20: Optimal Testing Accuracy:", metrics.f1_score(twenty_test.target, pred, average='macro'))

gsearch2 = GridSearchCV(LRpip2, param_grid=params, cv=5)
gsearch2.fit(imdb_train.data, imdb_train.target)
print(gsearch2.best_params_)
print(gsearch2.best_score_)
pred = gsearch2.predict(imdb_test.data)
print("(Required) imdb: Optimal Testing Accuracy:", metrics.f1_score(imdb_test.target, pred, average='macro'))
