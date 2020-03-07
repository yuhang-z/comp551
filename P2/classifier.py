# Author: Yuhang (7, Mar)
# compile environment: python3.8
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Import procedure
twenty_train = fetch_20newsgroups(subset='train', remove=(['headers','footers', 'quotes']), shuffle=True)

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


# Extracting features from text files (occurrences)
#
# Problem with occurrence counting: 
# longer documents will have higher average count values than shorter documents, 
# even though they might talk about the same topics.
#
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# #print(X_train_counts.shape)

# # Extracting features from text files (frequencey)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)

# # Training
# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# # shall be replaced
# docs_new = ['God is love', 'OpenGL on the GPU is fast', 'I love bitch']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicted = clf.predict(X_new_tfidf)

# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))

# Pipeline & Test
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




# Test 
twenty_test = fetch_20newsgroups(subset='test', remove=(['headers','footers', 'quotes']), shuffle=True)

# Logistic regression (0.6736%)
#logisticR(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# Decision tree (0.4048%)
#decisionT(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# SVC (0.6919%)
#SVC(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# AdaBoost (0.3747%)
#adaB(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)

# random forest (0.5754%)
#randomF(twenty_train.data, twenty_train.target, twenty_test.data, twenty_test.target)