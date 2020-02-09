
import numpy as np
import math
from numpy.linalg import inv
from dataProcess import ionosphere_builder
from dataProcess import adult_builder
from dataProcess import breastCancer_builder
from dataProcess import bank_builder


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
	inum = 100
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
	

	def selectTrainSets(self):
		return self.Xdata[0:self.sizeTrain, :], self.Ytarget[0:self.sizeTrain]	


	def fit(self, X, Y, learningRate, iterNum, epsilon):

		### Analytical Method
		# Direct Solution: w*=((X.T)(X))^(-1)(X.T)(Y)
		# Compute X.T dot X
		product1 = np.dot(X.T, X)
		# Compute X.T dot Y
		product2 = np.dot(X.T, Y)
		# Compute w*
		try:
			wstar = inv(product1).dot(product2)
		except np.linalg.LinAlgError as err:
			for i in range(len(product1)):
				product1[i][i] += 0.000001
			wstar = inv(product1).dot(product2)
		### Gradient-Descent Method
		### Uncomment the following lines for use
		# Call gradientDescent method 
		# wstar = self.gradientDescent(X, Y, learningRate, iterNum, epsilon)

		return wstar;


	def predict(self):

		### Do prediction on TEST sets
		# Yhat = sigma((wstar.T)(X))

		Xfortrain, Yfortrain = self.selectTrainSets()
		wstar = self.fit(Xfortrain, Yfortrain, self.lrate, self.inum, self.eps)
		designMatrix = np.dot(self.Xtest, wstar)
		
		for index in range(len(designMatrix)):
			logit = designMatrix[index]
			self.Yhattest.append(self.logistic_function(logit))

		Yresult = self.thresholding(self.Yhattest);
		accuracy = self.evaluate_acc(Yresult, self.Ytest)

		### Uncomment to see results
		# print (Yresult)
		# print (self.Ytest)
		print(accuracy)
		return accuracy


	def predictAccuracy_kfold(self, Xtrain, Ytrain, Xvalidation, Yvalidation):

		wstar = self.fit(Xtrain, Ytrain, self.lrate, self.inum, self.eps)
		designMatrix = np.dot(Xvalidation, wstar)
		yht = []
		for index in range(len(designMatrix)):
			logit = designMatrix[index]
			yht.append(self.logistic_function(logit))

		Yresult = self.thresholding(yht);
		accuracy = self.evaluate_acc(Yresult, Yvalidation)
		return accuracy


	def kfoldCrossValidation(self, k):

		numRow, numCol = self.Xdata.shape
		size_kfoldValidation = int((self.sizeTrain+self.sizeValidation)/k)
		
		#print(self.sizeTest)
		X_excludeTest = self.Xdata[0:numRow-self.sizeTest+1, :]
		Y_excludeTest = self.Ytarget[0:numRow-self.sizeTest+1]
		#print(Y_excludeTest.shape)
		#print(Y_excludeTest)

		t_accuracy = 0;
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

			t_accuracy = t_accuracy + self.predictAccuracy_kfold(Xtrain, Ytrain, Xvalidation, Yvalidation)

		print(t_accuracy/k)
		return t_accuracy/k



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


	def truthTable(Yhat, Y):
		
		### Helper Method to Evaluate Accuracy
		### EXTRA FEATURE: Positive & Negative
		for index in range(len(Yhat)):
			if (Yhat[index]>=0.5) and (Y[index]==1):
				self.tp = self.tp + 1;
			if (Yhat[index]<0.5) and (Y[index]==0):
				self.tn = self.tn + 1;
			if (Yhat[index]>=0.5) and (Y[index]==0):
				self.fp = self.fp + 1;
			if (Yhat[index]<0.5) and (Y[index]==1):
				self.fn = self.fn + 1;


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
				Yresult.append(1)
			else:
				Yresult.append(0)

		return Yresult

	def logistic_function(self, logit):
		return 1.0 / (1.0 + math.e**(-logit))



		
