from os import error
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

theta = [0.0, 0.0]
try:
    fil = open("theta_result.txt", "r")
except Exception as ex:
    print("NO TRAIN MODEL")
    exit()
theta[0] = float(fil.readline().rstrip('\n'))
theta[1] = float(fil.readline())
fil.close()

kmt = float(sys.argv[1])
if (kmt < 0 or kmt > 390000):
    print("INVALID KM - TOO LARGE OR TOO SMALL")
    exit()

def ft_min_max_norm(minmax, num):
    return(((num - minmax['min']) / (minmax['max'] - minmax['min'])))

def ft_min_max_denorm(minmax, num):
    return(num * (minmax['max'] - minmax['min']) + minmax['min'])

def predict_price(theta, norm_km):
	return theta[0] + theta[1] * norm_km
try:
    df = pd.read_csv('data.csv')
except Exception as ex:
    print("NO TRAIN DATA")
    exit()

km = np.array(df.iloc[:,:-1].values, dtype='float64')
price = np.array(df.iloc[:,1].values, dtype='float64')

minmaxKm = {'min': float(min(km)), 'max': float(max(km))}
minmaxPr = {'min': min(price), 'max': max(price)}

p=ft_min_max_denorm(minmaxPr, predict_price(theta, ft_min_max_norm(minmaxKm, kmt)))
print(kmt, "km =", p)
plt.scatter(km,price,color = 'red')
plt.scatter(kmt,p,color = 'blue')
plt.xlabel("km")
plt.ylabel("price")
plt.show()