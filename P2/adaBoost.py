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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


### Specify pipeline
adapip = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', AdaBoostClassifier(n_estimators=200, learning_rate=0.8, random_state=0)),
])


print("Default Parameters: n_estimators=200, learning_rate=0.8")

# ### PART I (Possible Bonus): Perform Training on training set, Predictions also on training set
adapip.fit(twenty_train.data, twenty_train.target)
pred = adapip.predict(twenty_train.data)
print("(Bonus) 20: Training Set Accuracy:", metrics.f1_score(twenty_train.target, pred, average='macro'))

adapip.fit(imdb_train.data, imdb_train.target)
pred = adapip.predict(imdb_train.data)
print("(Bonus) imdb: Training Set Accuracy:", metrics.f1_score(twenty_train.target, pred, average='macro'))


# ### PART II (Required): Perform Training on training set, Predictions on test set
pred = adapip.predict(twenty_test.data)
print("(Required) 20: Test Set Accuracy:", metrics.f1_score(twenty_test.target, pred, average='macro'))

pred = adapip.predict(imdb_test.data)
print("(Required) imdb: Test Set Accuracy:", metrics.f1_score(twenty_test.target, pred, average='macro'))


### Part III (Required): K-Fold cross validation
print("(Required) 20: K-cv score before tuning:", cross_val_score(adapip, twenty_train.data, twenty_train.target, cv=5, scoring='accuracy').mean())

print("(Required) imdb: K-cv score before tuning:", cross_val_score(adapip, imdb_train.data, imdb_train.target, cv=5, scoring='accuracy').mean())
# kf = KFold(n_splits=5, random_state=None, shuffle=False)
# print(twenty_train.data.shape)
# for train_index, test_index in kf.split(twenty_train):
# 	adapip.fit(twenty_train.data[train_index], twenty_train.target[test_index])
# 	pred = adapip.predict(twenty_train.data[test_index])
# 	print(metrics.f1_score(twenty_train.target[test_index], pred, average='macro'))

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
adapip.fit(comp_train.data, comp_train.target)
pred = adapip.predict(comp_test.data)
print("(Bonus) Comp. Test Set Accuracy:", metrics.f1_score(comp_test.target, pred, average='macro'))
adapip.fit(rec_train.data, rec_train.target)
pred = adapip.predict(rec_test.data)
print("(Bonus) Rec. Test Set Accuracy:", metrics.f1_score(rec_test.target, pred, average='macro'))
adapip.fit(sci_train.data, sci_train.target)
pred = adapip.predict(sci_test.data)
print("(Bonus) Sci. Test Set Accuracy:", metrics.f1_score(sci_test.target, pred, average='macro'))
adapip.fit(talk_train.data, talk_train.target)
pred = adapip.predict(talk_test.data)
print("(Bonus) Talk. Test Set Accuracy:", metrics.f1_score(talk_test.target, pred, average='macro'))


### Part V (Required): Parameter tuning using kcv

# # k-fold validation after tuning using random search
# params = {
#     "clf__n_estimators": [50, 100],
#     "clf__learning_rate": [0.01, 0.05, 0.1, 0.3, 1],
#     #"clf__loss": ['linear', 'square', 'exponential'],
#     "clf__algorithm": ['SAMME', 'SAMME.R']
#     }

# turned_dt_random = RandomizedSearchCV(adapip, param_distributions=params, cv=5)

# turned_dt_random.fit(twenty_train.data, twenty_train.target)
# best_estimator = turned_dt_random.best_estimator_

# print('Best n_estimators(random search):', turned_dt_random.best_estimator_.get_params()['clf__n_estimators'])
# print('Best learning_rate(random search):', turned_dt_random.best_estimator_.get_params()['clf__learning_rate'])
# #print('Best loss(random search):', turned_dt_random.best_estimator_.get_params()['clf__loss'])
# print('Best algorithm(random search):', turned_dt_random.best_estimator_.get_params()['clf__algorithm'])

# y_estimated = turned_dt_random.predict(twenty_test.data)
# acc = np.mean(y_estimated == twenty_test.target)
# print("Accuracy after tuning:{}".format(acc))
