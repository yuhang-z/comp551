
import numpy as np
import math
from numpy.linalg import inv
from dataProcess import ionosphere_builder


class Logistic_Regression:

	# Data Matrix X
	Xdata = [[]]
	# Target Matrix Y
	Ytarget = [[]]
	# Member Viariables for evalation
	tp = 0
	tn = 0
	fp = 0
	fn = 0


	lrate = 0.1
	inum = 500
	eps = 0.01

	# MAY BE USEFUL
	Yhat = []


	def __init__(self, dataname):
		
		self.dataname = dataname
		# Load the datasets into NumPy objects
		self.Xdata,self.Ytarget = ionosphere_builder()
		# Add Bias (concatenate a 1 to x)
		self.Xdata = np.insert(self.Xdata, 0, 1, axis=1)
		# TODO: Randomize the datasets


	def fit(self, X, Y, learningRate, iterNum, epsilon):

		### Direct Solution: w*=((X.T)(X))^(-1)(X.T)(Y)
		# Compute X.T dot X
		product1 = np.dot(X.T, X)
		# Compute X.T dot Y
		product2 = np.dot(X.T, Y)
		# Compute w*
		wstar = inv(product1).dot(product2)

		### Call gradientDescent
		self.gradientDescent(X, Y, learningRate, iterNum, epsilon)

		return wstar;


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

	
	def stochastic_gradientDescent(X, Y, learningRate, iterNum, epsilon):
		
		### EXTRA FEATURE: Stochastic Gradient-Descent
		# Refrence: Lecture 6 "SGD for logistic Regresssion"
		N,D = X.shape
		w = np.zeros(D)
		g = np.inf
		count = 0
		
		while (np.linalg.norm(g)>epsilon) and (count<iterNum):
			n = np.random.randint(N)
			g = gradient(X[[n],:], Y[[n]], w)
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
				tp = tp + 1;
			if (Yhat[index]<0.5) and (Y[index]==0):
				tn = tn + 1;
			if (Yhat[index]>=0.5) and (Y[index]==0):
				fp = fp + 1;
			if (Yhat[index]<0.5) and (Y[index]==1):
				fn = fn + 1;


	def predict(self):

		### Do prediction on TEST sets
		# Yhat = sigma((wstar.T)(X))

		# TODO: select sets for training 
		Xfortrain, Yfortrain = self.selectTrainSets()
		Xfortest = self.selectTestSets()
		wstar = self.fit(Xfortrain, Yfortrain, self.lrate, self.inum, self.eps)
		designMatrix = np.dot(Xfortest, wstar)
		
		#print(designMatrix)
		
		for index in range(len(designMatrix)):
			logit = designMatrix[index]
			self.Yhat.append(self.logistic_function(logit))
			# print yhat[index][0]
		print(self.Yhat)
		print(self.Ytarget[310:320])


	def selectTrainSets(self):
		#print(self.Xdata[0:300, :])
		#print(self.Ytarget[0:300])
		return self.Xdata[0:300, :], self.Ytarget[0:300]


	def selectTestSets(self):
		return self.Xdata[310:320, :]


	def logistic_function(self, logit):
		return 1.0 / (1.0 + math.e**(-logit))


	def kfcValidation(data_X,data_y,fold):
		total_rows = np.size(data_X,0)
		total_accuracy = 0
		for i in range(1,fold):

			# prepare training data and validation data for  k fold

			vali_first_row = int(((i-1)/fold) * total_rows)
			vali_last_row = int((i/fold) * total_rows)
			vali_X = data_X[num_first_row,:]
			vali_y = data_y[num_first_row,:]

			for j in range(vali_first_row,vali_last_row):
				vali_X = np.row_stack((vali_X,data_X[j,:]))
				vali_y = np.row_stack((vali_y,data_y[j,:]))

				train_X =  data_X[0,:]
				train_y =  data_Y[0,:]

				if i == 1 :
					train_X = data_X[vali_last_row+1:,]
					train_y = data_y[vali_last_row+1:,]
				elif i == k :
					train_X  = data_X[0:vali_first_row,:]
					train_y  = data_y[0:vali_first_row,:]
				else :
					train_X = np.row_stack(( data_X[0:vali_first_row,:],data_X[vali_last_row+1:,:] ))
					train_y = np.row_stack(( data_y[0:vali_first_row,:],data_y[vali_last_row+1:,:] ))

					train_X = np.delete(train_X,0,0)
					train_y = np.delete(train_X,0,0)

					# use train xy, vali xy to get accuracy
					# accuracy = ### call Function with train X ,train Y ,val_X and Val_y
					# total_accuracy = total_accuracy + accuracy

		return (total_accuracy/fold)


	def evaluate_acc():
		truthTable(Yhat, Y)
