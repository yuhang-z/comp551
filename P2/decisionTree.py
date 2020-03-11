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
from sklearn.tree import DecisionTreeClassifier


### Specify pipeline
DTpip = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', DecisionTreeClassifier()),
])


print("Default Parameters: Sklearn Defaults. The nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")

# ### PART I (Possible Bonus): Perform Training on training set, Predictions also on training set
DTpip.fit(twenty_train.data, twenty_train.target)
pred = DTpip.predict(twenty_train.data)
print("(Bonus) Training Set Accuracy:", metrics.f1_score(twenty_train.target, pred, average='macro'))

# ### PART II (Required): Perform Training on training set, Predictions on test set
pred = DTpip.predict(twenty_test.data)
print("(Required) Test Set Accuracy:", metrics.f1_score(twenty_test.target, pred, average='macro'))


### Part III (Required): K-Fold cross validation
print("(Required) K-cv score before tuning:", cross_val_score(DTpip, twenty_train.data, twenty_train.target, cv=5, scoring='accuracy').mean())
# kf = KFold(n_splits=5, random_state=None, shuffle=False)
# print(twenty_train.data.shape)
# for train_index, test_index in kf.split(twenty_train):
#   DTpip.fit(twenty_train.data[train_index], twenty_train.target[test_index])
#   pred = DTpip.predict(twenty_train.data[test_index])
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
DTpip.fit(comp_train.data, comp_train.target)
pred = DTpip.predict(comp_test.data)
print("(Bonus) Comp. Test Set Accuracy:", metrics.f1_score(comp_test.target, pred, average='macro'))
DTpip.fit(rec_train.data, rec_train.target)
pred = DTpip.predict(rec_test.data)
print("(Bonus) Rec. Test Set Accuracy:", metrics.f1_score(rec_test.target, pred, average='macro'))
DTpip.fit(sci_train.data, sci_train.target)
pred = DTpip.predict(sci_test.data)
print("(Bonus) Sci. Test Set Accuracy:", metrics.f1_score(sci_test.target, pred, average='macro'))
DTpip.fit(talk_train.data, talk_train.target)
pred = DTpip.predict(talk_test.data)
print("(Bonus) Talk. Test Set Accuracy:", metrics.f1_score(talk_test.target, pred, average='macro'))



# # k-fold validation after tuning using random search
# params = {
#     "clf__max_depth": [400],
#     "clf__max_features": [None, 'auto', 'log2'],
#     "clf__criterion": ["gini", "entropy"],
#     "clf__min_samples_split": [14, 15, 16, 17, 18, 19, 20, 21, 22],
#     "clf__min_samples_leaf": [12, 14, 16, 17, 18, 19, 20, 22, 24],
#     "clf__class_weight": ["balanced", None]
#     }

# turned_dt_random = RandomizedSearchCV(DTpip, param_distributions=params, cv=5)

# turned_dt_random.fit(twenty_train.data, twenty_train.target)
# best_estimator = turned_dt_random.best_estimator_

# print('Best max_features(random search):', turned_dt_random.best_estimator_.get_params()['clf__max_features'])
# print('Best criterion(random search):', turned_dt_random.best_estimator_.get_params()['clf__criterion'])
# print('Best min_samples_split(random search):', turned_dt_random.best_estimator_.get_params()['clf__min_samples_split'])
# print('Best min_samples_leaf(random search):', turned_dt_random.best_estimator_.get_params()['clf__min_samples_leaf'])
# print('Best class_weight(random search):', turned_dt_random.best_estimator_.get_params()['clf__class_weight'])

# y_estimated = turned_dt_random.predict(twenty_test.data)
# acc = np.mean(y_estimated == twenty_test.target)
# print("Accuracy after tuning:{}".format(acc))
