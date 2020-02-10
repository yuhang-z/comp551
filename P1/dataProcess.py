#author: Yuhang Zhang & Oliva Xu & Diyang Zhang
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

    # load iono_y form iono_matrix and encode it
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

    iono_X = np.asarray(iono_X)
    iono_y = np.asarray(iono_y)
    return iono_X, iono_y

#================== matrix bilder for adult.txt ===========================
def adult_builder():
    global adult_X, adult_y
    adult_matrix = genfromtxt('adult.txt', dtype='unicode', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (range(15)))
    
    # shuffle matrix 
    np.random.shuffle(adult_matrix)

    # load adult_X from iono_matrix 
    adult_X = adult_matrix[0:len(adult_matrix)-1,0:14].copy()

    # load adult_y form iono_matrix and encode it
    adult_y = encode_X(adult_matrix[0:len(adult_matrix)-1,14].copy())

    #Integer Encode of each column
    numRow, numCol = adult_X.shape
    for i in range(numRow):
        for j in range(numCol):
            element = adult_X[i][j]
            if not re.match("^[0-9 ]+$", element):
                adult_X[i][j] = stoi(element)
            else:
                adult_X[i][j] = int(element)

    helper_X = []
    for i in range(numRow):
        tmp = []
        for j in range(numCol):
            tmp.append(int(adult_X[i][j]))
        helper_X.append(tmp)

    helper_X = np.asarray(helper_X)
    adult_y = np.asarray(adult_y)
    # print(helper_X)
    # print(bank_y)

    return helper_X, adult_y


#================== matrix bilder for bank.txt ============================
def bank_builder():
    global bank_X, bank_y
    bank_matrix = genfromtxt('bank.txt', dtype='unicode', delimiter=';', filling_values = 'MAL_SAMPLE', usecols = (range(17)))

    bank_matrix = bank_matrix[1: , :]
    # shuffle matrix 
    np.random.shuffle(bank_matrix)

    # load bank_X from iono_matrix 
    bank_X = bank_matrix[0:len(bank_matrix)-1,0:16].copy()

    # load bank_y form iono_matrix and encode it
    bank_y = encode_X(bank_matrix[0:len(bank_matrix)-1,16].copy())

    numRow, numCol = bank_X.shape
    for i in range(numRow):
        for j in range(numCol):
            element = bank_X[i][j]
            if not re.match("^[0-9 ]+$", element):
                bank_X[i][j] = stoi(element)
            else:
                bank_X[i][j] = int(element)

    helper_X = []
    for i in range(numRow):
        tmp = []
        for j in range(numCol):
            tmp.append(int(bank_X[i][j]))
        helper_X.append(tmp)

    helper_X = np.asarray(helper_X)
    bank_y = np.asarray(bank_y)
    # print(helper_X)
    # print(bank_y)

    return helper_X, bank_y

#================== matrix bilder for breast-cancer.txt =====================
def breastCancer_builder():
    global cancer_X, cancer_y

    cancer_matrix = genfromtxt('breast-cancer.txt', dtype='unicode', delimiter=',', filling_values = 'MAL_SAMPLE', usecols = (range(10)))

    # shuffle matrix 
    np.random.shuffle(cancer_matrix)

    # load cancer_X from iono_matrix 
    cancer_X = cancer_matrix[0:len(cancer_matrix)-1,0:9].copy()

    # load cancer_y form iono_matrix and encode it
    cancer_y = encode_X(cancer_matrix[0:len(cancer_matrix)-1,9].copy())


    numRow, numCol = cancer_X.shape
    for i in range(numRow):
        for j in range(numCol):
            element = cancer_X[i][j]
            if not re.match("^[0-9 ]+$", element):
                cancer_X[i][j] = stoi(element)
            else:
                cancer_X[i][j] = int(element)

    helper_X = []
    for i in range(numRow):
        tmp = []
        for j in range(numCol):
            tmp.append(int(cancer_X[i][j]))
        helper_X.append(tmp)

    helper_X = np.asarray(helper_X)
    cancer_y = np.asarray(cancer_y)
    # print(helper_X)
    # print(cancer_y)

    return helper_X, cancer_y


### Helper function to assign a value to a string
def stoi(str):
    result = 0
    for i in range(len(str)):
        result = result + ord(str[i])
    return result

#=========================== Test funtions ================================
#Test funtions: The underlining functions shall be used in the main function 

#ionosphere_builder()
#adult_builder()
#bank_builder()
#breastCancer_builder()
