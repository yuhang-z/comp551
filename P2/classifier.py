# Author: Yuhang (7, Mar)
# compile environment: python3.8

# Libraries: 
import numpy as np
from pprint import pprint
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier



# Basics of sklearn dataset formating:

# 1. "target_names": class names 
#print(twenty_train.target_names)
#
#print(twenty_train.target_names[twenty_train.target[0]])

# 2. "data": actual data of dataset
#print("\n".join(twenty_train.data[0].split("\n")[:4]))

# 3. "target": 
#print(twenty_train.target[:40])
#
#for t in twenty_train.target[:10]:
#    print(twenty_train.target_names[t])



# Import 20 groups dataset
twenty_train = fetch_20newsgroups(subset='train', remove=(['headers','footers', 'quotes']), shuffle=True)


# Import IMDB reviews dataset
IMDb_train = datasets.load_files("/Users/YuhangZhang/desktop/gh/comp551/P2/IMDB_dataset", description=None, categories=None, load_content=True, shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)

# check if data is successfully loaded:
# "target_names": class names 
#print(IMDb_train.target_names)



# Pipeline & Test of 20 groups
def logisticR(trainData, trainTarget, testData, testTarget):

    text_lr = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression()),
    ])

    text_lr.fit(trainData, trainTarget)

    predicted = text_lr.predict(testData)
    print(np.mean(predicted == testTarget))


def decisionT(trainData, trainTarget, testData, testTarget):

    text_dt = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', DecisionTreeClassifier(random_state=0)),
    ])

    text_dt.fit(trainData, trainTarget)

    predicted = text_dt.predict(testData)
    print(np.mean(predicted == testTarget))


def SVC(trainData, trainTarget, testData, testTarget):

    text_svc = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC(random_state=0, tol=1e-5)),
    ])

    text_svc.fit(trainData, trainTarget)

    predicted = text_svc.predict(testData)
    print(np.mean(predicted == testTarget))


def adaB(trainData, trainTarget, testData, testTarget):

    text_ada = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', AdaBoostClassifier(random_state=0)),
    ])

    text_ada.fit(trainData, trainTarget)

    predicted = text_ada.predict(testData)
    print(np.mean(predicted == testTarget))


def randomF(trainData, trainTarget, testData, testTarget):

    text_rf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(max_depth=30, random_state=0)),
    ])

    text_rf.fit(trainData, trainTarget)

    predicted = text_rf.predict(testData)
    print(np.mean(predicted == testTarget))


# Test of 20 groups
twenty_test = fetch_20newsgroups(subset='test', remove=(['headers','footers', 'quotes']), shuffle=True)

# Logistic regression (0.6736)
#logisticR(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# Decision tree (0.4048)
#decisionT(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# SVC (0.6919)
#SVC(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# AdaBoost (0.3747)
#adaB(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# random forest (0.5754)
#randomF(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)


# Test of IMDb
IMDb_test = datasets.load_files("/Users/YuhangZhang/desktop/gh/comp551/P2/IMDB_dataset", description=None, categories=None, load_content=True, shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)

# Logistic regression (0.93328)
#logisticR(IMDb_train.data, IMDb_train.target, IMDb_test.data, IMDb_test.target)

# Decision tree (1.0)
#decisionT(IMDb_train.data, IMDb_train.target, IMDb_test.data, IMDb_test.target)

# SVC (0.98996)
#SVC(IMDb_train.data, IMDb_train.target, IMDb_test.data, IMDb_test.target)

# AdaBoost (0.80996)
#adaB(IMDb_train.data, IMDb_train.target, IMDb_test.data, IMDb_test.target)

# random forest (0.97652)
#randomF(IMDb_train.data, IMDb_train.target, IMDb_test.data, IMDb_test.target)


# "Validation" using k-fold cross validation

def logisticR_validation(trainData, trainTarget):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainData)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    kfold = KFold(n_splits=5, shuffle=False)
    #Logistic Regression supports only penalties in ['l1', 'l2', 'elasticnet', 'none']
    model = LogisticRegression()
    results = cross_val_score(model, X_train_tfidf, trainTarget, cv=kfold)
    print(results)
    print("Accuracy:", results.mean())


def decisionT_validation(trainData, trainTarget):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainData)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    kfold = KFold(n_splits=5, shuffle=False)
    #Logistic Regression supports only penalties in ['l1', 'l2', 'elasticnet', 'none']
    model = DecisionTreeClassifier(random_state=0)
    results = cross_val_score(model, X_train_tfidf, trainTarget, cv=kfold)
    print(results)
    print("Accuracy:", results.mean())


def SVC_validation(trainData, trainTarget):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainData)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    kfold = KFold(n_splits=5, shuffle=False)
    #Logistic Regression supports only penalties in ['l1', 'l2', 'elasticnet', 'none']
    model = LinearSVC(random_state=0, tol=1e-5)
    results = cross_val_score(model, X_train_tfidf, trainTarget, cv=kfold)
    print(results)
    print("Accuracy:", results.mean())


def adaB_validation(trainData, trainTarget):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainData)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    kfold = KFold(n_splits=5, shuffle=False)
    #Logistic Regression supports only penalties in ['l1', 'l2', 'elasticnet', 'none']
    model = AdaBoostClassifier(random_state=0)
    results = cross_val_score(model, X_train_tfidf, trainTarget, cv=kfold)
    print(results)
    print("Accuracy:", results.mean())


def randomF_validation(trainData, trainTarget):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainData)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    kfold = KFold(n_splits=5, shuffle=False)
    #Logistic Regression supports only penalties in ['l1', 'l2', 'elasticnet', 'none']
    model = RandomForestClassifier(max_depth=30, random_state=0)
    results = cross_val_score(model, X_train_tfidf, trainTarget, cv=kfold)
    print(results)
    print("Accuracy:", results.mean())


# Data set 1, 20 groups =========================
# Logistic regression (Accuracy: 0.7193744913463932)
#logisticR_validation(twenty_train.data, twenty_train.target)

# Decision tree (Accuracy: 0.4395444260941693)
#decisionT_validation(twenty_train.data, twenty_train.target)

# SVC (Accuracy: 0.7596785719448648)
#SVC_validation(twenty_train.data, twenty_train.target)

# AdaBoost (Accuracy: 0.39764879448850987)
#adaB_validation(twenty_train.data, twenty_train.target)

# random forest (Accuracy: 0.6119837324615846)
#randomF_validation(twenty_train.data, twenty_train.target)


# Data set 2, IMDb =========================
# Logistic regression (Accuracy: 0.8878)
#logisticR_validation(IMDb_train.data, IMDb_train.target)

# Decision tree (Accuracy: 0.7052000000000002)
#decisionT_validation(IMDb_train.data, IMDb_train.target)

# SVC (Accuracy: 0.8925599999999999)
#SVC_validation(IMDb_train.data, IMDb_train.target)

# AdaBoost (Accuracy: 0.80472)
#adaB_validation(IMDb_train.data, IMDb_train.target)

# random forest (Accuracy: 0.8301999999999999)
#randomF_validation(IMDb_train.data, IMDb_train.target)



#TODO: auto hyperparameter generator is not yet successful
# Turing hyperparameter using "random Search"
def turningLogR(trainData, trainTarget):
    penalty = ['l1', 'l2']
    C = np.linspace(1,200)

    hyperparams = dict(C=C, penalty=penalty)
    model = LogisticRegression()

    rsearch = RandomizedSearchCV(model, hyperparams, n_iter=100, random_state=41)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(trainData)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    rsearch.fit(X_train_tfidf, trainTarget)

    print(rsearch.best_params_)


# hyperparameter tuning
#turningLogR(IMDb_train.data, IMDb_train.target)

