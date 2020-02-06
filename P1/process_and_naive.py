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
from pandas import read_csv
from numpy import set_printoptions
import pandas as pd

#=========================== Helper Functions =============================

#===================== General OneHotEncode function ======================
def encode(data):
    values = array(data)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    #** Not required **
    # binary encode
    #onehot_encoder = OneHotEncoder(sparse = False)
    #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    #onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(integer_encoded)
    return integer_encoded

    #** Not required **
    # invert first example ** Not required **
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    #print(inverted)

#===== 4 individual matrix-builder as each dataset has few in common ======

#================== matrix bilder for ionosphere.txt ======================
def ionosphere_builder():
    global iono_X, iono_y
    iono_X = genfromtxt('ionosphere.txt', dtype='float32', delimiter=',', filling_values = -99, usecols = (range(34)))
    #print(iono_X.shape)

    # Remove malsamples
    for i in range(len(iono_X)):
        for j in range(len(iono_X[i-1])):
            if iono_X[i-1][j] == -99 :
                iono_X = np.delete(iono_X, i-1, 0)
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

    adult_X = np.delete(adult_X, [7, 8, 11, 12], 1)

    #Delete malsamples
    list_to_delete=[]
    for i in range(len(adult_X)):
        for j in range(len(adult_X[i-1])):
            if adult_X[i-1][j] == 'MAL_SAMPLE' or adult_X[i-1][j]==b' ?':
                #adult_X = np.delete(adult_X, i-1, 0)
                list_to_delete.append(i-1)
    #print(list_to_delete)
    adult_X = np.delete(adult_X, list_to_delete, 0)
    #print(adult_X)
    #print(adult_X.shape)

#=========================== Test funtions ================================
#Test funtions: The underlining functions shall be used in the main function

#ionosphere_builder()
adult_builder()

def bank_builder():
    global bank_X, bank_y
    bank_X = genfromtxt('bank.txt', dtype='S20', delimiter=';', skip_header=1, usecols = np.arange(0,15))
    bank_y = genfromtxt('bank.txt', dtype='S8', delimiter=';',  usecols = (16))
    bank_y = np.delete(bank_y,0,0)
    #print(len(bank_X))
    #print(bank_X[0])
    #print(len(bank_y))
    #print(bank_y[0])

    #print(bank_X.shape)

    #print(bank_X[0:3])
    #print(bank_y[0])



bank_builder()

def breast_builder():
    global breast_X, breast_y
    breast_X = genfromtxt('breast-cancer.txt', dtype='S20', delimiter=',', usecols = np.arange(0,9))
    breast_y = genfromtxt('breast-cancer.txt', dtype='S8', delimiter=',',  usecols = (9))
    #print(len(breast_X))
    #print(len(breast_y))

    print(breast_X[285])
    print(breast_y[285])
    print(breast_X[272])
    print(breast_y[272])
    print(breast_X[282])
    print(breast_y[282])




    #print(breast_X[0:3])


    #print(breast_X.shape)

breast_builder()


########################### --- Naive Bayes --- ##############################

#prior class probability
def prior(label,Y_data):
    num_result = 0

    for i in range(len(Y_data)):
        if label ==  Y_data[i]:
            num_result= num_result + 1

    return num_result/len(Y_data)

#print(adult_y[0])
#print(prior(b' <=50K',adult_y))

#likelihood of input features given the class label

def likelihood(label,features,X_data,Y_data):
    list_of_label = []
    for i in range(len(Y_data)):
        if Y_data[i]==label:
            list_of_label.append(i)
    X_datasubset = X_data[list_of_label,:]
    #print(X_datasubset)
    #print(X_datasubset.shape)
    all = len(X_datasubset)+1
    multiply_result= 1
    of_the_feature = 0

    for i in range(len(features)):
        feature = features[i]

        for j in range(len(X_datasubset)):
            if X_datasubset[j,i]==feature:
                of_the_feature = of_the_feature+1

        term_to_multiply = of_the_feature/all
        #print('term')
        #rint(term_to_multiply)
        of_the_feature = 0
        multiply_result = multiply_result*term_to_multiply

    return multiply_result

#fea = [b'no-recurrence-events',b'30-39',b'premeno',b'30-34',b'0-2',b'no',b'3',b'left',b'left_up']
#fea=[b'recurrence-events',b'50-59',b'ge40',b'30-34',b'3-5',b'no',b'3',b'left',b'left_low']
#print('test')

#marginal probability of the input
def evidence(features,X_data,Y_data,label1,label2):
    p_label1=prior(label1,Y_data)
    p_label2=prior(label2,Y_data)

    likelihood_label1=likelihood(label1,features,X_data,Y_data)
    likelihood_label2=likelihood(label2,features,X_data,Y_data)
    #print(p_label1)
    #print(p_label2)
    #print(likelihood_label1)
    #print(likelihood_label2)
    result = p_label1*likelihood_label1 + p_label2*likelihood_label2
    return result


breast_train_data_X = (np.array(breast_X))
breast_train_data_X = breast_train_data_X[1:269]
print(breast_train_data_X.shape)
breast_train_data_Y = (np.array(breast_y))
breast_train_data_Y = breast_train_data_Y[1:269]


#print(breast_X.shape)
##print(breast_train_data_X.shape)
#print(breast_train_data_Y.shape)
#print(breast_train_data_X[2])
#print(breast_X[2])

#label1 predict probability
def naive(input_features,train_X,train_Y,label1,label2):
    pc = prior(label1,train_Y)
    pxc = likelihood(label1,input_features,train_X,train_Y)
    px = evidence(input_features,train_X,train_Y,label1,label2)
    #print('pxc')
    #print(pxc)
    if px == 0:
        return 0

    return ((pc*pxc)/px)

###############- test breast_cancer

f_270 = [b'recurrence-events',b'50-59',b'ge40',b'30-34',b'3-5',b'no',b'3',b'left',b'left_low']
p_breast_270_yes = naive(f_270,breast_train_data_X,breast_train_data_Y,b'yes',b'no')
print(p_breast_270_yes)

f_272 = [b'recurrence-events',b'40-49',b'premeno',b'15-19',b'0-2',b'yes',b'3',b'right',b'left_up']
p_breast_272_yes = naive(f_272,breast_train_data_X,breast_train_data_Y,b'yes',b'no')
print(p_breast_272_yes)

f_283 = [b'recurrence-events',b'30-39',b'premeno',b'20-24',b'0-2',b'no',b'3',b'left',b'left_up']
p_breast_283_yes = naive(f_283,breast_train_data_X,breast_train_data_Y,b'yes',b'no')
print(p_breast_283_yes)
