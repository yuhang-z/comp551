#author: Yuhang Zhang & Oliva Xu 
#This class contains functions that build and analysis dataset samples.
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

#=========================== 2 Helper Functions: ==========================
#===================== Encode X string variables into integers ============
def encode_X(data):
    values = array(data)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    return integer_encoded


#========= Encode y result into bianry numbers **ONE HOT ENCODING** =======

def encode_y(data):
    values = array(data)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded



#===== 4 individual matrix-builder as each dataset has few in common ======
#================== matrix bilder for ionosphere.txt ======================
def ionosphere_builder():
    global iono_X, iono_y
    iono_X = genfromtxt('ionosphere.txt', dtype='float32', delimiter=',', filling_values = -99, usecols = (range(34)))
    iono_y = genfromtxt('ionosphere.txt', dtype='S8', delimiter=',', filling_values = 'g', usecols = (34))
    print(iono_X.shape)

    # Remove malsamples
    for i in range(len(iono_X)):
        for j in range(len(iono_X[i-1])):
            if iono_X[i-1][j] == -99 :
                iono_X = np.delete(iono_X, i-1, 0)

    # Remove feature at column 1 
    iono_X = np.delete(iono_X, 1, 1)
    

#================== matrix bilder for adult.txt ===========================
def adult_builder():
    global adult_X, adult_y
    adult_X = genfromtxt('adult.txt', dtype='S20', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (range(14)))
    adult_y = genfromtxt('adult.txt', dtype='S8', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (14))
    #print(adult_X.shape)

    #Integer Encode of each column
    for i in range(len(adult_X[0])):
        if not any(char.isdigit() for char in adult_X[0][i]):
            data = adult_X[:, i]
            adult_X[:, i] = encode_X(data)

    #Delete malfeatures
    adult_X = np.delete(adult_X, [7, 8, 11, 12], 1)

    #Delete malsamples 
    for i in range(len(adult_X)):
        for j in range(len(adult_X[i-1])):
            if adult_X[i-1][j] == 'MAL_SAMPLE' :
                adult_X = np.delete(adult_X, i-1, 0)

    #ONE HOT ENCODING of y 
    adult_y = encode_y(adult_y)


#================== matrix bilder for bank.txt ============================
def bank_builder():
    global bank_X, bank_y
    bank_X = genfromtxt('bank.txt', dtype='S20', delimiter=';', filling_values = 'MAL_SAMPLE', usecols = np.arange(0,15))
    bank_y = genfromtxt('bank.txt', dtype='S2', delimiter=';',  usecols = (16))

    #Integer Encode of each column
    for i in range(len(bank_X[0])):
        if not any(char.isdigit() for char in bank_X[0][i]):
            data = bank_X[:, i]
            bank_X[:, i] = encode_X(data)

    #ONE HOT ENCODING of y 
    bank_y = encode_y(bank_y)

    #Delete malsamples 
    for i in range(len(bank_X)):
        for j in range(len(bank_X[i-1])):
            if bank_X[i-1][j] == 'MAL_SAMPLE' :
                bank_X = np.delete(bank_X, i-1, 0)

#================== matrix bilder for breast-cancer.txt =====================
def breastCancer_builder():
    global cancer_X, cancer_y
    cancer_X = genfromtxt('breast-cancer.txt', dtype='S20', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = np.arange(0,9))
    cancer_y = genfromtxt('breast-cancer.txt', dtype='S8', delimiter=',',  usecols = (9))

    #Integer Encode of each column
    for i in range(len(cancer_X[0])):
        if not any(char.isdigit() for char in cancer_X[0][i]):
            data = cancer_X[:, i]
            cancer_X[:, i] = encode_X(data)

    #ONE HOT ENCODING of y 
    cancer_y = encode_y(cancer_y)

    #Delete malsamples 
    for i in range(len(cancer_X)):
        for j in range(len(cancer_X[i-1])):
            if cancer_X[i-1][j] == 'MAL_SAMPLE' :
                cancer_X = np.delete(cancer_X, i-1, 0)

#=========================== Test funtions ================================
#Test funtions: The underlining functions shall be used in the main function 

#ionosphere_builder()
#adult_builder()
#bank_builder()
#breastCancer_builder()