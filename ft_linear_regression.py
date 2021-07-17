import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ft_min_max_norm(minmax, num):
	return(((num - minmax['min']) / (minmax['max'] - minmax['min'])))

def ft_min_max_denorm(minmax, num):
	return(num * (minmax['max'] - minmax['min']) + minmax['min'])

def predict_price(theta, norm_km):
	return theta[0] + theta[1] * norm_km

def gradient_des(n, theta, norm_km, norm_price, theta_tmp, cost_t, error_cost):
	for i in range(n):
		y_pred = predict_price(theta, norm_km[i])   # predict value for given x
		error_cost += (norm_price[i] - y_pred)**2
		theta_tmp[0] += (y_pred - norm_price[i])
		theta_tmp[1] += ((y_pred - norm_price[i]) * norm_km[i])
		cost_t[0] += theta_tmp[0] * (-2)
		cost_t[1] += theta_tmp[1] * (-2)
	return(theta_tmp, error_cost)

def main ():

	#import dataset	
	df = pd.read_csv('data.csv')

	km = np.array(df.iloc[:,:-1].values, dtype='float64')
	price = np.array(df.iloc[:,1].values, dtype='float64')
	n = np.size(km)

	minmaxKm = {'min': float(min(km)), 'max': float(max(km))}
	minmaxPr = {'min': min(price), 'max': max(price)}
	norm_km = ft_min_max_norm(minmaxKm, km)
	norm_price = ft_min_max_norm(minmaxPr, price)

	theta = [0.0,0.0]
	lr = 0.1
	epoches = 4000
	error = []
	regr = [[],[]]

	for epoch in range(epoches):
		error_cost = 0
		cost_t = [0.0, 0.0]
		theta_tmp = [0.0, 0.0]
		theta_tmp, error_cost = gradient_des(n, theta, norm_km, norm_price, theta_tmp, cost_t, error_cost)
		theta[0] -= lr * (theta_tmp[0] / n)
		theta[1] -= lr * (theta_tmp[1] / n)
		if(epoch % 100 == 0):
			print(epoch,theta[0],theta[1])
			regr[0].append(epoch)
			regr[1].append(predict_price(theta,norm_km))
		error.append(error_cost)
	
	error1 = norm_price - predict_price(theta,norm_km).T
	se = np.sum(error1 ** 2)
	mse = se/float(n)

	print("mean squared error is", mse)
	print("Theta0 = ", theta[0])
	print("Theta1 = ", theta[1])

	with open("theta_result.txt", "w") as file:
		file.write(str(float(theta[0])) + '\n' + str(float(theta[1])))

if __name__ == "__main__":
	main()