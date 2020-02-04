
import numpy as np
import math
from numpy.linalg import inv


class Logistic_Regression:

	# Data Matrix X
	Xdata = []
	# Target Matrix Y
	Ytarget = []
	# Member Viariables for evalation
	tp = 0
	tn = 0
	fp = 0
	fn = 0



	# MAY BE USEFUL
	Yhat = [[]]



	def __init__(self, dataname):
    	
    	self.dataname = dataname
    	# Load the datasets into NumPy objects
    	loadData(self.Xdata, self.Ytarget)
    	
    	# TODO: Randomize the datasets


	def fit(X, Y, learningRate, iterNum, eps):
		
		### Direct Solution: w*=((X.T)(X))^(-1)(X.T)(Y)
		# Add Bias (concatenate a 1 to x)
		Xb = np.insert(X, 0, 1, axis=1)
		# Compute X.T dot X
		product1 = np.dot(Xb.T, Xb)
		# Compute X.T dot Y
		product2 = np.dot(Xb.T, Y)
		# Compute w*
		wstar = inv(product1).dot(product2)

		### Call gradientDescent
		self.gradientDescent()

		return wstar;


	def gradientDescent(X, Y, learningRate, eps, iterNum=1000):
		
		### Gradient-Descent
		# Refrence: Lecture 6 "GD for logistic Regresssion"
		N,D = X.shape
		w = np.zeros(D)
		g = np.inf
		count = 0
		
		while (np.linalg.norm(g)>eps) && (count<iterNum):
			g = gradient(X, Y, w)
			w = w - learningRate*g
			count++

		return w

	
	def stochastic_gradientDescent(X, Y, learningRate, eps, iterNum=1000)
		
		### EXTRA FEATURE: Stochastic Gradient-Descent
		# Refrence: Lecture 6 "SGD for logistic Regresssion"
		N,D = X.shape
		w = np.zeros(D)
		g = np.inf
		count = 0
		
		while (np.linalg.norm(g)>eps) && (count<iterNum):
			n = np.random.randint(N)
			g = gradient(X[[n],:], Y[[n]], w)
			w = w - learningRate*g
			count++

		return w
	

	def gradient(X, Y, w):

		### Helper Method to Compute Gradient
		# Refrence: Lecture 6 "GD for logistic Regresssion"
		N,D = X.shape
		yh = np.dot(X, W)
		grad = np.dot(X.T, yh-y) / N
		return grad

	def truthTable(Yhat, Y):
		
		### Helper Method to Evaluate Accuracy
		### EXTRA FEATURE: Positive & Negative
		for index in range(len(Yhat)):
			if (Yhat[index][0]>=0.5) && (Y[index][0]==1):
				tp++;
			if (Yhat[index][0]<0.5) && (Y[index][0]==0):
				tn++;
			if (Yhat[index][0]>=0.5) && (Y[index][0]==0):
				fp++;
			if (Yhat[index][0]<0.5) && (Y[index][0]==1):
				fn++;


	def predict(self, Xfortest):

		### Do prediction on TEST sets
		# Yhat = sigma((wstar.T)(X))

		# TODO: select sets for training 

		wstar = self.fit(Xfortrain, Yfortrain, learningRate, iterNum, eps)
		designMatrix = np.dot(Xfortest, wstar)
		for index in range(len(designMatrix)):
			logit = designMatrix[index][0]
			Yhat[index][0] = self.logistic_function(logit)
			# print yhat[index][0]		


	def logistic_function(logit):
		return 1.0 / (1.0 + math.e**(-logit))


	def kCrossValidation():

	def evaluate_acc():
		truthTable(Yhat, Y)

	def selectTestsets(self):
