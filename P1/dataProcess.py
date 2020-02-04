#author: Yuhang Zhang
#This class contains functions that build and analysis dataset samples.
#Sklearn shall be installed: https://calebshortt.com/2016/01/15/installing-scikit-learn-python-data-mining-library/

#=================================imports==================================
import math
import csv
import numpy as np
import sklearn
from numpy import array
from numpy import argmax
from numpy import genfromtxt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#=========================== Helper Functions =============================

#===================== General OneHotEncode function ======================
def encode(data):
    values = array(data)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

    #** Not required **
    # invert first example ** Not required **
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    #print(inverted)

#===== 4 individual matrix-builder as each dataset has few in common ======

#================== matrix bilder for ionosphere.txt ======================
def ionosphere_builder():
    global iono_X, iono_y
    iono_X = genfromtxt('ionosphere.txt', dtype='float32', delimiter=',', filling_values = -99, usecols = (range(34)))
    print(iono_X.shape)

    # Remove malsamples
    for i in range(len(iono_X)):
        for j in range(len(iono_X[i])):
            if iono_X[i][j] == -99 :
                iono_X = np.delete(iono_X, i, 0)
    #print(iono_X.shape)

    # Remove feature at column 1 
    iono_X = np.delete(iono_X, 1, 1)
    iono_y = genfromtxt('ionosphere.txt', dtype='S8', delimiter=',', filling_values = 'g', usecols = (34))
    
    #TEST1
    #print(iono_X)[1][3]
    #print(iono_y)

#================== matrix bilder for adult.txt ===========================
def adult_builder():
    global adult_X, adult_y
    adult_X = genfromtxt('adult.txt', dtype='S20', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (range(14)))
    adult_y = genfromtxt('adult.txt', dtype='S8', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (14))
    #print(adult_X.shape)

    #Delete malfeatures
    adult_X = np.delete(adult_X, [7, 8, 11, 12], 1)

    #One Hot Encode of each column

    for i in range(len(adult_X[0])):
        if not adult_X[0][i].isdigit():
            data = adult_X[:, i]
            #print(data)
            print(encode(data))

    #adult_X = finalmatrix\
    #print(adult_X.shape)

    #Delete malsamples 
    for i in range(len(adult_X)):
        for j in range(len(adult_X[i])):
            if adult_X[i][j] == 'MAL_SAMPLE' :
                adult_X = np.delete(adult_X, i, 0)
    #print(adult_X)
    #print(adult_X.shape)

    
    #print(adult_X)
    #print(adult_X.shape)




#ionosphere_builder()
adult_builder()