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
    iono_X = genfromtxt('ionosphere.txt', dtype='float32', delimiter=',', filling_values = '0.', usecols = (range(34)))
    # Remove feature at column 1 
    iono_X = np.delete(iono_X, 1, 1)
    iono_y = genfromtxt('ionosphere.txt', dtype='S8', delimiter=',', filling_values = 'g', usecols = (34))
    
    #TEST1
    #print(iono_X)[1][3]
    #print(iono_y)

#================== OneHotEncode for adult.txt ============================ 
#Column 1 workclass
adult_col1 = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov']
result_adult_col1 = encode(adult_col1)
#print(result_adult_col1)

#Column 3 education
adult_col3 = ['Bachelors', 'HS-grad', 'Masters', 'Some-college', 'Doctorate', 'Assoc', 'Prof-school', 'Other']
result_adult_col3 = encode(adult_col3)
#print(result_adult_col3)

#Column 6 marital-status
adult_col5 = ['Never-married', 'Married', 'Divorced', 'Separated']
result_adult_col5 = encode(adult_col5)
#print(result_adult_col5)

#Column 8 race
adult_col8 = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
result_adult_col8 = encode(adult_col8)
#print(result_adult_col8)

#Column 9 sex
adult_col9 = ['Male', 'Female']
result_adult_col9 = encode(adult_col9)
#print(result_adult_col9)

#Column 13
adult_col13 = ['United-States', 'Other']
result_adult_col13 = encode(adult_col13)
#print(result_adult_col13)

#Column y
adult_y = ['<=50K', '>50K']
result_adult_y = encode(adult_y)
#print(result_adult_y)

#================== matrix bilder for adult.txt ===========================
def adult_builder():
    global adult_X, adult_y
    adult_X = genfromtxt('adult.txt', dtype='S20', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (range(14)))
    #print(adult_X.shape)

    #Delete malsamples 
    for i in range(len(adult_X)):
        for j in range(len(adult_X[i])):
            if adult_X[i][j] == 'MAL_SAMPLE' :
                adult_X = np.delete(adult_X, i, 0)
    print(adult_X)
    #print(adult_X.shape)

    #Delete malfeatures
    adult_X = np.delete(adult_X, [7, 8, 11, 12], 1)
    print(adult_X)
    #print(adult_X.shape)


#ionosphere_builder()
adult_builder()