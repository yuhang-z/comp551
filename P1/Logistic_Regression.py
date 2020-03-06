### Author: Yuhang Zhang
###         Olivia Xu
###	    Diyang Zhang

import numpy as np
import math
from numpy.linalg import inv
from dataProcess import ionosphere_builder
from dataProcess import adult_builder
from dataProcess import breastCancer_builder
from dataProcess import bank_builder
np.seterr(divide='ignore',invalid='ignore')

# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression


class Logistic_Regression:

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

	### Hyper-parameters
	# Learning Rate
	lrate = 0.01
	# Iteration Number
	inum = 500
	# Epsilon
	eps = 1e-2

	# Distribution of original datasets
	sizeTrain = 0
	sizeValidation = 0
	sizeTest = 0

	# Testsets
	Xtest = [[]]
	Ytest = []
	Yhattest = []

	# Label for Analytical or GD method
	label = ''


	def __init__(self, dataname, label):
		
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
		self.label = label; 
		# Add Bias (concatenate a 1 to x)
		self.Xdata = np.insert(self.Xdata, 0, 1, axis=1)
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


	def fit(self, X, Y, learningRate, iterNum, epsilon):

		
		### FULL Batch Gradient Descent
		if self.label == 'G':
			wstar = self.gradientDescent(X, Y, learningRate, iterNum, epsilon)
			# wstar = self.stochastic_gradientDescent(X, Y, learningRate, iterNum, epsilon)

		if self.label == 'T':
			wstar = self.gradientDescent(X, Y, learningRate, iterNum, epsilon)

		return wstar;

	def fit_extra(self, X, Y):

		#==========================EXTRA FEATURE=========================#
		#========================Analytical Method=======================#
		
		if self.label == 'A':
			
			x1, y1, x0, y0 = [], [], [], []
			for i in range(len(Y)):
				if Y[i]==1:
					x1.append(X[i, :])
					y1.append(1)
				else:
					x0.append(X[i, :])
					y0.append(0)
			x1 = np.asarray(x1)
			y1 = np.asarray(y1)
			x0 = np.asarray(x0)
			y0 = np.asarray(y0)

			# Compute X1.T dot X1
			x1x1 = np.dot(x1.T, x1)
			# Compute X1.T dot Y1
			x1y1 = np.dot(x1.T, y1)
			# Compute w1*
			w1_star = np.linalg.pinv(x1x1).dot(x1y1)

			# Compute X0.T dot X0
			x0x0 = np.dot(x0.T, x0)
			# Compute X0.T dot Y0
			x0y0 = np.dot(x0.T, y0)
			# Compute w1*
			w0_star = np.linalg.pinv(x0x0).dot(x0y0)

			# designMatrix0 = np.dot(X, w0_star)
			# designMatrix1 = np.dot(X, w1_star)
			# for i in range(len(X)):
			# 	print(designMatrix0[i], designMatrix1[i])
			return np.subtract(w1_star, w0_star)


	def predict(self):

		### Do prediction on TEST sets
		# Yhat = sigma((wstar.T)(X))
		if self.label == 'A':
			Xfortrain, Yfortrain = self.Xdata[0:int(self.sizeTrain/4), :], self.Ytarget[0:int(self.sizeTrain/4)]
			wstar = self.fit_extra(Xfortrain, Yfortrain)
			designMatrix = np.dot(self.Xtest, wstar)
			yht = []
			for index in range(len(designMatrix)):
				logit = designMatrix[index]
				yht.append(self.logistic_function(logit))
			# print(yht)

		
		elif self.label == 'G':
			
			tmp = int(self.sizeTrain/4)
			Xfortrain, Yfortrain = self.Xdata[0:tmp*3, :], self.Ytarget[0:tmp*3]
			# Xfortrain, Yfortrain = self.Xdata[0:self.sizeTrain, :], self.Ytarget[0:self.sizeTrain]
			wstar = self.fit(Xfortrain, Yfortrain, self.lrate, self.inum, self.eps)
			designMatrix = np.dot(self.Xtest, wstar)
			yht = []
			for index in range(len(designMatrix)):
				yht.append(designMatrix[index])
		
<<<<<<< HEAD
		## Thresholding the result
		Yresult = self.thresholding(yht)
=======
		# Thresholding the result
		# Yresult = self.thresholding_extra(yht);
		Yresult = self.thresholding(yht);
>>>>>>> ab8cd4bafe0a87504adf367c15e7e3e148088641

		accuracy = self.evaluate_acc(Yresult, self.Ytest)

		# ### Uncomment to see results
		# print (Yresult)
		# print (self.Ytest)
		print(accuracy)
		return accuracy


	def predictAccuracy_kfold(self, Xtrain, Ytrain, Xvalidation, Yvalidation):

		if self.label == 'A':
			wstar = self.fit_extra(Xtrain, Ytrain)
			designMatrix = np.dot(Xvalidation, wstar)
			yht = []
			for index in range(len(designMatrix)):
				logit = designMatrix[index]
				yht.append(self.logistic_function(logit))
			# print(yht)

		elif self.label == 'G':
			wstar = self.fit(Xtrain, Ytrain, self.lrate, self.inum, self.eps)
			designMatrix = np.dot(Xvalidation, wstar)
			yht = []
			for index in range(len(designMatrix)):
				yht.append(designMatrix[index])

<<<<<<< HEAD
		Yresult = self.thresholding(yht)
=======
		# Yresult = self.thresholding(yht);
		Yresult = self.thresholding_extra(yht);
		# print(yht)
		# print(Yresult)
>>>>>>> ab8cd4bafe0a87504adf367c15e7e3e148088641
		accuracy = self.evaluate_acc(Yresult, Yvalidation)
		# print(Yresult)
		# print(Yvalidation)
		# print()
		return accuracy


	def kfoldCrossValidation(self, k):

		numRow, numCol = self.Xdata.shape
		size_kfoldValidation = int((self.sizeTrain+self.sizeValidation)/k)
		
		#print(self.sizeTest)
		X_excludeTest = self.Xdata[0:numRow-self.sizeTest+1, :]
		Y_excludeTest = self.Ytarget[0:numRow-self.sizeTest+1]
		#print(Y_excludeTest.shape)
		#print(Y_excludeTest)

<<<<<<< HEAD
		t_accuracy = 0
=======
		t_accuracy = 0;
		t_trainAccuracy = 0;
>>>>>>> ab8cd4bafe0a87504adf367c15e7e3e148088641
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

			
			# t_trainAccuracy = t_trainAccuracy + self.predictAccuracy_kfold(Xtrain, Ytrain, Xtrain, Ytrain)
			# t_accuracy = t_accuracy + self.predictAccuracy_kfold(Xtrain, Ytrain, Xvalidation, Yvalidation)

			# t_trainAccuracy = t_trainAccuracy + self.predictAccuracy_kfold(Xtrain, Ytrain, Xtrain, Ytrain)
			t_accuracy = t_accuracy + self.predictAccuracy_kfold(Xtrain[0:int(len(Xtrain)/4*2), :], Ytrain[0:int(len(Xtrain)/4*2)], Xvalidation, Yvalidation)



		# print("Train Acc", t_trainAccuracy/k)	

		print("Validation Acc:", t_accuracy/k)
		
		# return t_accuracy/k
		# return(t_trainAccuracy/k)



	def gradientDescent(self, X, Y, learningRate, iterNum, epsilon):
		
		### Gradient-Descent
		# Refrence: Lecture 6 "GD for logistic Regresssion"
		N,D = X.shape
		w = np.zeros(D)
		g = np.inf
		count = 0
		
		while (np.linalg.norm(g)>epsilon) and (count<iterNum):
			g = self.gradient(X, Y, w)
			w = w - learningRate*g
			count = count + 1

		# print(count)

		return w

	
	def stochastic_gradientDescent(self, X, Y, learningRate, iterNum, epsilon):
		
		### EXTRA FEATURE: Stochastic Gradient-Descent
		# Refrence: Lecture 6 "SGD for logistic Regresssion"
		N,D = X.shape
		w = np.zeros(D)
		g = np.inf
		count = 0
		
		while (np.linalg.norm(g)>epsilon) and (count<iterNum):
			n = np.random.randint(N)
			g = self.gradient(X[[n],:], Y[[n]], w)
			w = w - learningRate*g
			count = count + 1

		return w
	

	def gradient(self, X, Y, w):

		### Helper Method to Compute Gradient
		# Refrence: Lecture 6 "GD for logistic Regresssion"
		N,D = X.shape
		yh = np.dot(X, w)
		grad = np.dot(X.T, yh-Y) / N
		return grad


	# def truthTable(Yhat, Y):
		
	# 	### Helper Method to Evaluate Accuracy
	# 	### EXTRA FEATURE: Positive & Negative
	# 	for index in range(len(Yhat)):
	# 		if (Yhat[index]>=0.5) and (Y[index]==1):
	# 			self.tp = self.tp + 1
	# 		if (Yhat[index]<0.5) and (Y[index]==0):
	# 			self.tn = self.tn + 1
	# 		if (Yhat[index]>=0.5) and (Y[index]==0):
	# 			self.fp = self.fp + 1
	# 		if (Yhat[index]<0.5) and (Y[index]==1):
	# 			self.fn = self.fn + 1


	def evaluate_acc(self, Yresult, Y):

		### Calculate "Accuracy"
		count = 0
		for index in range(len(Yresult)):
			if (Yresult[index]==Y[index]):
				count = count + 1
		return count/len(Yresult)


	def thresholding(self, yh):

		### Thresholding Yhat by 0.5 to Yresult
		Yresult = []
		for singley in yh:
			if (singley>=0.5):
			# if (singley>=0.7):
				Yresult.append(1)
			else:
				Yresult.append(0)

		return Yresult

	def thresholding_extra(self, yh):

		Yresult = []
		for singley in yh:
			if (singley>=0.7):
			# if (singley>=0.7):
				Yresult.append(1)
			else:
				Yresult.append(0)

		return Yresult

	def logistic_function(self, logit):
		return 1.0 / (1.0 + math.e**(-logit))

