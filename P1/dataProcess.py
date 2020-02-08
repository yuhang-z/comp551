#author: Yuhang Zhang & Oliva Xu 
#This class contains functions that build and analysis dataset samples.
#=================================imports==================================
import math
import csv
import re
import numpy as np
import sklearn
from numpy import array
from numpy import argmax
from numpy import genfromtxt
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#=========================== 3 Helper Functions: ==========================
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

#========= data cleanning -> remove the blank space in adult.txt ==========

def clean(input):
    output = []
    for i in range(len(input)): 
        b_row = []
        for j in range(len(input[i])):
            b_row.append(input[i][j].strip())
        np.append(output, b_row, axis=0)
    return output         


#===== 4 individual matrix-builder as each dataset has few in common ======
#================== matrix bilder for ionosphere.txt ======================
def ionosphere_builder():
    global iono_X, iono_y
    iono_matrix = genfromtxt('ionosphere.txt', dtype='S20', delimiter=',', filling_values = -99, usecols = (range(35)))

    # shuffle matrix 
    np.random.shuffle(iono_matrix)

    # load iono_X from iono_matrix 
    iono_X = iono_matrix[0:len(iono_matrix)-1,0:34].copy()

    # load iono-y form iono_matrix and encode it
    iono_y = encode_X(iono_matrix[0:len(iono_matrix)-1,34].copy())

    # convert the datatype of X from "string" to "float"
    iono_X = iono_X.astype(float)

    # Remove malsamples
    for i in range(len(iono_X)):
        for j in range(len(iono_X[i-1])):
            if iono_X[i-1][j] == -99 :
                iono_X = np.delete(iono_X, i-1, 0)

    # Remove feature at column 1 
    iono_X = np.delete(iono_X, 1, 1)

    # Test of ionosphere_builder():
    print(iono_X)
    print(iono_X.shape)
    print(iono_y)
    print(iono_y.shape)
    
    return iono_X, iono_y

#================== matrix bilder for adult.txt ===========================
def adult_builder():
    global adult_X, adult_y
    adult_matrix = genfromtxt('adult.txt', dtype='S20', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (range(15)))
    
    # shuffle matrix 
    np.random.shuffle(adult_matrix)

    # load iono_X from iono_matrix 
    adult_X = adult_matrix[0:len(adult_matrix)-1,0:14].copy()

    # load iono-y form iono_matrix and encode it
    adult_y = encode_X(adult_matrix[0:len(adult_matrix)-1,14].copy())
    
    #Integer Encode of each column
    for i in range(len(adult_X[0])):
            if not re.match("^[0-9 ]+$", adult_X[0][i]):
                data = adult_X[:, i-1]
                #
                adult_X[:, i] = encode_X(data).astype(int)
            else:
                adult_X[:, i] = adult_X[:, i].astype(float)     

    #Delete malfeatures
    adult_X = np.delete(adult_X, [7, 8, 11, 12], 1)

    print(adult_X)
    print(adult_y)

    b = []
    print(adult_matrix.shape)
    for i in range(len(adult_matrix)):
        for j in range(len(adult_matrix[i])):
            if adult_matrix[i][j] == ' ?':
                b.append(i)
    
    for badrow in b:
        adult_matrix = np.delete(adult_matrix, badrow, 0)


    return adult_X, adult_y

#================== matrix bilder for bank.txt ============================
def bank_builder():
    global bank_X, bank_y
    bank_matrix = genfromtxt('bank.txt', dtype='S20', delimiter=';', filling_values = 'MAL_SAMPLE', usecols = (range(16)))

    # shuffle matrix 
    np.random.shuffle(bank_matrix)

    # load iono_X from iono_matrix 
    bank_X = bank_matrix[0:len(bank_matrix)-1,0:15].copy()

    # load iono-y form iono_matrix and encode it
    bank_y = encode_X(bank_matrix[0:len(bank_matrix)-1,15].copy())

    #Integer Encode of each column
    for i in range(len(bank_X[0])):
            if not re.match("^[0-9 ]+$", bank_X[0][i]):
                data = bank_X[:, i-1]
                #
                bank_X[:, i] = encode_X(data).astype(int)
            else:
                bank_X[:, i] = bank_X[:, i].astype(float)    

    #ONE HOT ENCODING of y 
    bank_y = encode_y(bank_y)

    return bank_X, bank_y

#================== matrix bilder for breast-cancer.txt =====================
def breastCancer_builder():
    global cancer_X, cancer_y

    cancer_matrix = genfromtxt('breast-cancer.txt', dtype='S20', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (range(10)))

    # shuffle matrix 
    np.random.shuffle(cancer_matrix)

    # load iono_X from iono_matrix 
    cancer_X = cancer_matrix[0:len(cancer_matrix)-1,0:9].copy()

    # load iono-y form iono_matrix and encode it
    cancer_y = encode_X(cancer_matrix[0:len(cancer_matrix)-1,9].copy())

    #Integer Encode of each column
    for i in range(len(cancer_X[0])):
            if not re.match("^[0-9 ]+$", cancer_X[0][i]):
                data = cancer_X[:, i-1]
                #
                cancer_X[:, i] = encode_X(data).astype(int)
            else:
                cancer_X[:, i] = cancer_X[:, i].astype(float)   
    print(cancer_X)
    #ONE HOT ENCODING of y 
    cancer_y = encode_y(cancer_y)

    return cancer_X, cancer_y

#=========================== Test funtions ================================
#Test funtions: The underlining functions shall be used in the main function 

#ionosphere_builder()
#adult_builder()
#bank_builder()
#breastCancer_builder()