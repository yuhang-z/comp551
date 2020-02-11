# Author: Yuhang Zhang & Olivia Xu & Diyang Zhang
# Naive Bayes Classifier is feasible here since we have filtered data samples in the way that features are mutually independent
# Assumptions:  1. matrix_X, and matrix_y
#               2. For matrix_X:
#                   - continous elements have type "float" 
#                   - discrete elements are encoded as "integer"
#               3. For matrix_y:
#                   - binary results are coded either integer "0", or "1" 
#                   - one demention list   
#=================================imports==================================
import math
import numpy as np
from math import sqrt
from math import exp
from math import pi
from dataProcess import ionosphere_builder
from dataProcess import adult_builder
from dataProcess import breastCancer_builder
from dataProcess import bank_builder
import scipy.stats

#=========================== Helper Functions: ============================

class Naive_Bayes:

    # Data Matrix X
    Xdata = [[]]
    # Target Matrix Y
    Ytarget = []
    
    ### Member Viariables for evalation
    # True-positive
    tp = 0
    # True-negative
    tn = 0
    # False-positive
    fp = 0
    # False-negative
    fn = 0

    # Distribution of original datasets
    sizeTrain = 0
    sizeValidation = 0
    sizeTest = 0

    # Testsets
    Xtest = [[]]
    Ytest = []
    Yhattest = []


    def __init__(self, dataname):
        
        self.dataname = dataname
        # Load the datasets into NumPy objects
        # NOTE: Data has been already randomized in dataProcess.py
        if dataname == "ionosphere":
            self.Xdata,self.Ytarget = ionosphere_builder()
        if dataname == "adult":
            self.Xdata,self.Ytarget = adult_builder()
        if dataname == "breast-cancer":
            self.Xdata,self.Ytarget = breastCancer_builder()
        if dataname == "bank":
            self.Xdata,self.Ytarget = bank_builder()

        # Dataset Distribution
        numRow, numCol = self.Xdata.shape
        # 80% for train (DEFAULT)
        self.sizeTrain = int(numRow*0.8)
        # 10% for Validation (DEFAULT)
        self.sizeValidation = int(numRow*0.1)
        # Last 10% for test (DEFAULT)
        self.sizeTest = numRow-self.sizeTrain-self.sizeValidation
        self.Xtest = self.Xdata[numRow-self.sizeTest:, :]
        self.Ytest = self.Ytarget[numRow-self.sizeTest:]


    # calculate the mean of a list of numbers
    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))

     
    # calculate the standard deviation of a list of numbers
    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = float(sum([(x-avg)**2 for x in numbers]) / float(len(numbers)))
        return float(sqrt(variance))


    # calculate the Gaussian probability distribution function for x
    def calculate_probability(self, x, mean, stdev):
        
        if not stdev == 0:
            exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
            return (1 / (sqrt(2 * pi) * stdev)) * exponent
        else:
            return 1.    
        '''
        if not stdev == 0:
            return scipy.stats.norm(mean, stdev).cdf(x)
        else: 
            return 1    
        '''

    #=========================== naiveBayesClassifer ==========================
    def fit(self, sample, matrix_X_train, matrix_y_train):
        
        # get size of col and row from matrix_X
        num_row = matrix_X_train.shape[0]
        num_col_X = matrix_X_train.shape[1]

        # Step_1
        # calculate the prior probability

        # count the number of results that equal to 0  
        count = 0
        for i in range(num_row):
            if matrix_y_train[i] == 0:
                count += 1
        p_prior_0 = float(count) / float(num_row)
        #print(count)
        #print(num_row)
        #print(p_prior_0)
        p_prior_1 = 1. - p_prior_0

        # Step_2
        # calculate the likelihood

        #likelihood variables shall be defined as float numbers 
        p_discrete_0, p_discrete_1, p_continous_0, p_continous_1 = 1., 1., 1., 1.


        for i in range(num_col_X):
            # discrete numbers ** categorical distribution **
            if isinstance(matrix_X_train[0][i], int):
                count_d_0, count_d_1 = 0, 0
                for j in range(num_row):
                    if matrix_X_train[j][i] == sample[i] and matrix_y_train[j] == 0:
                        count_d_0 += 1
                    if matrix_X_train[j][i] == sample[i] and matrix_y_train[j] == 1:   
                        count_d_1 += 1
                # accumulate p 
                p_discrete_0 = p_discrete_0 * (float(count_d_0)/float(num_row))   
                p_discrete_1 = p_discrete_1 * (float(count_d_1)/float(num_row))          
            # continous numbers ** gaussian distribution **
            else: 
                #load features according to result number
                nums_0, nums_1 = [], []
                for j in range(num_row):
                    if matrix_y_train[j] == 0:
                        nums_0.append(matrix_X_train[j][i])
                    else: 
                        nums_1.append(matrix_X_train[j][i])
                # mean & stdev at each column

                mean_0, stdev_0 = self.mean(nums_0), self.stdev(nums_0)
                mean_1, stdev_1 = self.mean(nums_1), self.stdev(nums_1)
                
                p_continous_0 = p_continous_0 * self.calculate_probability(sample[i], mean_0, stdev_0)
                p_continous_1 = p_continous_1 * self.calculate_probability(sample[i], mean_1, stdev_1)

        # Total prior * likelihood
        p_0 = p_prior_0 * p_discrete_0 * p_continous_0
        p_1 = p_prior_1 * p_discrete_1 * p_continous_1

        return p_0, p_1

    
    # Predict
    def predict(self):
        
        tmpSize = int(self.sizeTrain/4)

        # Xtrain, Ytrain = self.Xdata[0:self.sizeTrain, :], self.Ytarget[0:self.sizeTrain]
        Xtrain, Ytrain = self.Xdata[0:tmpSize*1, :], self.Ytarget[0:tmpSize*1]
        yresult = []
        for i in range(self.sizeTest):
            p_0, p_1 = self.fit(self.Xtest[i], Xtrain, Ytrain)
            if p_0>p_1:
                yresult.append(0)
            else:
                yresult.append(1)

        # Uncomment to see the result
        # print(ytest)
        # print(yresult)
        
        # Return: Accuracy
        acc = self.evaluate_acc(yresult, self.Ytest)
        print(acc)
        return acc


    # Evaluate Accuracy
    def evaluate_acc(self, Yresult, Y):

        ### Calculate "Accuracy"
        count = 0
        for index in range(len(Yresult)):
            if (Yresult[index]==Y[index]):
                count = count + 1
        return count/len(Yresult)


    # Helper Method for predicting accuracies for kfold-CV
    def predictAccuracy_kfold(self, Xtrain, Ytrain, Xvalidation, Yvalidation):

        yresult=[]
        for i in range(len(Xvalidation)):
            p_0, p_1 = self.fit(Xvalidation[i], Xtrain, Ytrain)
            if p_0>p_1:
                yresult.append(0)
            else:
                yresult.append(1)

        acc = self.evaluate_acc(yresult, Yvalidation)
        return acc


    # kfold CV
    def kfoldCrossValidation(self, k):

        numRow, numCol = self.Xdata.shape
        size_kfoldValidation = int((self.sizeTrain+self.sizeValidation)/k)
        
        #print(self.sizeTest)
        X_excludeTest = self.Xdata[0:numRow-self.sizeTest+1, :]
        Y_excludeTest = self.Ytarget[0:numRow-self.sizeTest+1]
        #print(Y_excludeTest.shape)
        #print(Y_excludeTest)

        t_accuracy = 0;
        t_trainAccuracy = 0;
        for index in range(k):
            startIndex = size_kfoldValidation*index
            endIndex = startIndex + size_kfoldValidation - 1
            Xvalidation = X_excludeTest[startIndex:endIndex+1, :]
            Yvalidation = Y_excludeTest[startIndex:endIndex+1]
            if index==k-1:
                Xvalidation = X_excludeTest[startIndex:, :]
                Yvalidation = Y_excludeTest[startIndex:]

            if index==0:
                Xtrain = X_excludeTest[endIndex+1:, :]
                Ytrain = Y_excludeTest[endIndex+1:]
            elif index==k-1:
                Xtrain = X_excludeTest[0:startIndex, :]
                Ytrain = Y_excludeTest[0:startIndex]
            else:
                Xtrain = np.row_stack(( X_excludeTest[0:startIndex, :], X_excludeTest[endIndex+1:, :]))
                Ytrain = np.concatenate(( Y_excludeTest[0:startIndex], Y_excludeTest[endIndex+1:]))

            # t_accuracy = t_accuracy + self.predictAccuracy_kfold(Xtrain, Ytrain, Xvalidation, Yvalidation)
            t_accuracy = t_accuracy + self.predictAccuracy_kfold(Xtrain[0:int(len(Xtrain)/4*2),:], Ytrain[0:int(len(Ytrain)/4*2)], Xvalidation, Yvalidation)

            t_trainAccuracy = t_trainAccuracy + self.predictAccuracy_kfold(Xtrain, Ytrain, Xtrain, Ytrain)

        print(t_accuracy/k)
        # print(t_trainAccuracy/k)
        return t_accuracy/k
        # return(t_trainAccuracy/k)
