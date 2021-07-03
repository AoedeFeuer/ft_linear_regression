import pandas as pd
import numpy as np

class FTLinReg():
	def __init__(self, iteration, learning_rate):
		self.learning_rate = learning_rate
		self.iteration = iteration

	def fit(self, km, price):
		self.m, self.n = km.shape
		self.thetha1 = np.zeros(self.n)
		self.thetha0 = 0
		self.km = km
		self.price = price

		for i in range(self.iteration):
			self.update_theta1()
		return self

	def update_thetha1(self):
		price_prediction = self.predict(self.km)

		d_thetha1 = (-2 / self.m) * (self.km.T).dot(self.price - price_prediction)
		d_thetha0 = (-2 / self.m) * np.sum(self.price - price_prediction)

		self.thetha1 = self.thetha1 - self.learning_rate * d_thetha1
		self.thetha0 = self.thetha0 - self.learning_rate * d_thetha0
		return self

	def predict (self, km):
		return km.dot(self.thetha1) + self.thetha0

def main ():

	#import dataset
	data_info = pd.read_csv("data.csv")
	km = data_info.iloc[:,:-1].values
	price = data_info.iloc[:,1].values

	#train
	model = FTLinReg(iteration = 100, learning_rate = 0.01)
	model.fit(km, price)

if __name__ == "__main__":
	main()