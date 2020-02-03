#author: Yuhang Zhang
#This class contains functions that build and analysis dataset samples
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
#==========================================================================

#===== 4 individual matrix-builder as each dataset has few in common ======

#================== matrix bilder for ionosphere.txt ======================
def ionosphere_builder():
    global iono_X, iono_y
    iono_X = genfromtxt('ionosphere.txt', dtype='float32', delimiter=',', filling_values = '0.', usecols = (range(34)))
    iono_y = genfromtxt('ionosphere.txt', dtype='S5', delimiter=',', filling_values = 'g', usecols = (34))
    
    print(iono_X)
    print(iono_y)

#================== OneHotEncode for adult.txt ============================ 
data1 = ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov']
values = array(data1)
print(values)
#================== matrix bilder for adult.txt ===========================
def adult_builder():
    global adult_X, adult_y


ionosphere_builder()
    