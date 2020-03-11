# Author: Yuhang, Diyang
# compile environment: python3.8

# Libraries: 

from dataLoading import *

import numpy as np
from pprint import pprint
from sklearn import datasets
from sklearn import metrics
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


print("Default Parameters: tol=1e-5")

### PART I (Possible Bonus): Perform Training on training set, Predictions also on training set
SVMpip.fit(twenty_train.data, twenty_train.target)
pred = SVMpip.predict(twenty_train.data)
print("(Bonus) 20: Training Set Accuracy:", metrics.f1_score(twenty_train.target, pred, average='macro'))

SVMpip.fit(imdb_train.data, imdb_train.target)
pred = SVMpip.predict(imdb_train.data)
print("(Bonus) imdb: Training Set Accuracy:", metrics.f1_score(twenty_train.target, pred, average='macro'))


### PART II (Required): Perform Training on training set, Predictions on test set
pred = SVMpip.predict(twenty_test.data)
print("(Required) 20: Test Set Accuracy:", metrics.f1_score(twenty_test.target, pred, average='macro'))

pred = SVMpip.predict(imdb_test.data)
print("(Required) imdb: Test Set Accuracy:", metrics.f1_score(twenty_test.target, pred, average='macro'))


### Part III (Required): K-Fold cross validation
print("(Required) 20: K-cv score before tuning:", cross_val_score(SVMpip, twenty_train.data, twenty_train.target, cv=5, scoring='accuracy').mean())

print("(Required) imdb: K-cv score before tuning:", cross_val_score(SVMpip, imdb_train.data, imdb_train.target, cv=5, scoring='accuracy').mean())
# kf = KFold(n_splits=5, random_state=None, shuffle=False)
# print(twenty_train.data.shape)
# for train_index, test_index in kf.split(twenty_train):
#   SVMpip.fit(twenty_train.data[train_index], twenty_train.target[test_index])
#   pred = SVMpip.predict(twenty_train.data[test_index])
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
SVMpip.fit(comp_train.data, comp_train.target)
pred = SVMpip.predict(comp_test.data)
print("(Bonus) Comp. Test Set Accuracy:", metrics.f1_score(comp_test.target, pred, average='macro'))
SVMpip.fit(rec_train.data, rec_train.target)
pred = SVMpip.predict(rec_test.data)
print("(Bonus) Rec. Test Set Accuracy:", metrics.f1_score(rec_test.target, pred, average='macro'))
SVMpip.fit(sci_train.data, sci_train.target)
pred = SVMpip.predict(sci_test.data)
print("(Bonus) Sci. Test Set Accuracy:", metrics.f1_score(sci_test.target, pred, average='macro'))
SVMpip.fit(talk_train.data, talk_train.target)
pred = SVMpip.predict(talk_test.data)
print("(Bonus) Talk. Test Set Accuracy:", metrics.f1_score(talk_test.target, pred, average='macro'))



# # k-fold validation after tuning using random search
# params = {
#     "clf__C": [1, 2, 3],
#     "clf__loss": ['hinge', 'squared_hinge'],
#     "clf__max_iter": [1000, 2000, 3000],
#     "clf__fit_intercept": [True, False],
#     "clf__multi_class": ['ovr', 'crammer_singer']
#     }

# turned_svm_random = RandomizedSearchCV(SVMpip, param_distributions=params, cv=5, verbose=10, random_state=42, n_jobs=-1)

# turned_svm_random.fit(twenty_train.data, twenty_train.target)
# best_estimator = turned_svm_random.best_estimator_

# print('Best C(random search):', turned_svm_random.best_estimator_.get_params()['clf__C'])
# print('Best loss(random search):', turned_svm_random.best_estimator_.get_params()['clf__loss'])
# print('Best max_iter(random search):', turned_svm_random.best_estimator_.get_params()['clf__max_iter'])
# print('Best fit_intercept(random search):', turned_svm_random.best_estimator_.get_params()['clf__fit_intercept'])
# print('Best multi_class(random search):', turned_svm_random.best_estimator_.get_params()['clf__multi_class'])

# y_estimated = turned_svm_random.predict(twenty_test.data)
# acc = np.mean(y_estimated == twenty_test.target)
# print("Accuracy after tuning:{}".format(acc))
