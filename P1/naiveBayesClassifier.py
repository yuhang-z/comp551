# Author: Yuhang Zhang 
# Naive Bayes Classifier is feasible here since we has filter data samples in the way that features are independent
# Assumption:   1. matrix_X, and matrix_y
#               2. For matrix_X:
#                   continous elements have type "float" 
#                   discrete elements are encoded as "integer"
#               4. For matrix_y:
#                   binary results are coded either integer "0", or "1"    
#=================================imports==================================
import math
import numpy as np
from math import sqrt
from math import exp
from math import pi
#=========================== Helper Functions: ============================
# calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers) / float(len(numbers))
 
# calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

# calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

#=========================== naiveBayesClassifer ==========================
def nbc(sample, matrix_X, matrix_y):
    # get size of col and row
    num_row = matrix_X.shape[0]
    num_col_X = matrix_X.shape[1]

    # calculate the prior probability
    count = 0
    for i in range(num_row):
        if matrix_y[i] == 0:
            count += 1
    p_prior_0 = count / num_row
    p_prior_1 = 1 - p_prior_0

    # calculate the likelihood
    p_discrete_0, p_discrete_1, p_continous_0, p_continous_1 = 1, 1, 1, 1

    for i in range(num_col_X):
        # categorical distribution
        if isinstance(num_col_X[0][i], int):
            count_d_0, count_d_1 = 0, 0
            for j in range(num_row):
                if matrix_X[j][i] == sample[i] and matrix_y[j] == 0:
                    count_d_0 += 1
                if matrix_X[j][i] == sample[i] and matrix_y[j] == 1:   
                    count_d_1 += 1
            # accumulate p 
            p_discrete_0 = p_discrete_0 * (count_d_0/num_row)    
            p_discrete_1 = p_discrete_1 * (count_d_1/num_row)         
        else: # continous numbers 
            nums_0, nums_1 = [], []
            for j in range(num_row):
                if matrix_y[j] == 0:
                    nums_0.append(matrix_X[j][i])
                else: 
                    nums_1.append(matrix_X[j][i])
            # mean & stdev at each column
            mean_0, stdev_0 = mean(nums_0), stdev(nums_0)
            mean_1, stdev_1 = mean(nums_1), stdev(nums_1)
            #Total continous probability 
            p_continous_0 = p_continous_0 * calculate_probability(matrix_y[i], mean_0, stdev_0)
            p_continous_1 = p_continous_1 * calculate_probability(matrix_y[i], mean_1, stdev_1) 

    # Total prior * likelihood 
    p_0 = p_prior_0 * p_discrete_0 * p_continous_0
    p_1 = p_prior_1 * p_discrete_1 * p_continous_1

    if p_0 > p_1:
        print(0)     
    else:
        print(1)    