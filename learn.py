import numpy as np
import pandas as pd

#neural network class definition
class network:
	
	#getting the file name
	def __init__(self,file):
		self.file=file
		
	
	#loading dataset from file using pandas
	def load(self):
		dataset = pd.read_csv(self.file,sep=',',header=None)
		#extracting the inputs and adding bias to it
		self.X = np.insert(dataset.iloc[:,:-1].values,0,1,axis=1)
		#extracting the outputs and reshaping it to a column vector
		self.Y = dataset.iloc[:,-1].values.reshape(self.X.shape[0],1)
		
	
	#sigmoid function
	def __sigmoid(self,z):
		return 1/(1+np.exp(-z))
	
	
	#sigmoid derivative function
	def __sigmoid_derivative(self,z):
		return self.__sigmoid(z)*(1-self.__sigmoid(z))
		
	
	#training the network using gradient descent to get appropriate weights
	def train(self,epochs):
		#initiating the weights with random numbers in range (-1,1)
		np.random.seed(1)
		self.w_1 = 2*np.random.rand(3,2)-1
		np.random.seed(1)
		self.w_2 = 2*np.random.rand(3,1)-1
		#number of iterations of gradient descent
		for epoch in range(epochs):
			#calculating output for current values of weights by propagating forward
			z_1 = np.matmul(self.X,self.w_1)
			a_1 = np.insert(self.__sigmoid(z_1),0,1,axis=1)
			z_2 = np.matmul(a_1,self.w_2)
			a_2 = self.__sigmoid(z_2)
			#calculating errors using backward propagation
			error_2 = a_2-self.Y
			error_1 = np.delete(np.matmul(error_2,np.transpose(self.w_2)),0,axis=1)
			#updating weights
			self.w_2 = self.w_2-np.matmul(np.transpose(a_1),error_2*self.__sigmoid_derivative(z_2))
			self.w_1 = self.w_1-np.matmul(np.transpose(self.X),error_1*self.__sigmoid_derivative(z_1))
	
	
	#predicting output using trained weights by forward propagation
	def predict(self):
		z_1 = np.matmul(self.X,self.w_1)
		a_1 = np.insert(self.__sigmoid(z_1),0,1,axis=1)
		z_2 = np.matmul(a_1,self.w_2)
		a_2 = self.__sigmoid(z_2)
		print(self.file.strip('.txt').upper()+' output:')
		print(np.round(a_2))

