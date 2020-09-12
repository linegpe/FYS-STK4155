from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed

from sklearn.model_selection import train_test_split
import scipy.linalg as scl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
#from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from statistical_functions import *
from data_processing import *
from print_and_plot import *


#Polynomial degree, number of columns in the design matrix and number of data points
PolyDeg = 5
N = 20

#Setting x and y arrays
x = np.arange(0, 1, 1.0/N)
y = np.arange(0, 1, 1.0/N)



#x = np.sort(np.random.uniform(0,1,N))
#y = np.sort(np.random.uniform(0,1,N))

x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Adding noise and raveling
seed = 2022
#seed = 1402
alpha = 0.01
z_noise = AddNoise(z,seed,alpha,N)
z_noise_flat = np.matrix.ravel(z_noise)
x_flat = np.ravel(x)
y_flat = np.ravel(y)

# Scaling data
x_scaled, y_scaled, z_scaled, scaler = DataScaling(x_flat,y_flat,z_noise_flat)

# Set up design matrix and scale the data
X = DesignMatrix(x_scaled,y_scaled,PolyDeg)
X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)

#Beta from the OLS
beta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(z_train))
#beta = ols_svd(X_train,z_train)

# What happens here?
z_tilde = X_train.dot(beta)
z_predict = X_test.dot(beta)
z_plot = X.dot(beta)
z_plot_split = np.zeros((N,N))
z_plot_split = np.reshape(z_plot, (N,N))
x_rescaled, y_rescaled, z_rescaled = DataRescaling(x_scaled,y_scaled,z_plot,scaler,N)

# Print error estimations and plot the surfaces
PrintErrors(z_train,z_tilde,z_test,z_predict)
SurfacePlot(x,y,z_noise,z_rescaled)

#Variance-bias tradeoff
NumBootstraps = 100
MaxPolyDeg = 12

ModelComplexity =np.arange(0,MaxPolyDeg+1)

MSError_train = np.zeros(MaxPolyDeg+1)
Bias_train =np.zeros(MaxPolyDeg+1)
Variance_train =np.zeros(MaxPolyDeg+1)

MSError_test= np.zeros(MaxPolyDeg+1)
Bias_test =np.zeros(MaxPolyDeg+1)
Variance_test =np.zeros(MaxPolyDeg+1)

MSError_train_boot = np.zeros(MaxPolyDeg+1)

for i in range(MaxPolyDeg+1):
	X_new = DesignMatrix(x_scaled,y_scaled,i)
	#z_noise_pred = np.empty((len(z_noise[0]), NumBootstraps))
	model = LinearRegression(fit_intercept=False)

	X_train, X_test, z_train, z_test = train_test_split(X_new, z_scaled, test_size=0.2)

	z_predict_train = np.empty((len(z_train), NumBootstraps))
	z_predict_test = np.empty((len(z_test), NumBootstraps))
	MSE_boot = np.zeros(NumBootstraps)
	I = np.identity(int((i+2)*(i+1)/2))
	#z_test_new = z_test[:,np.newaxis]

	for j in range(NumBootstraps):
		X_, z_ = resample(X_train, z_train)
		beta_ridge = np.linalg.pinv(X_.T.dot(X_)+0.01*I).dot(X_.T.dot(z_))
		z_predict_test[:,j] = X_test.dot(beta_ridge)
		z_predict_train[:,j] = X_train.dot(beta_ridge)
		MSE_boot[j] = MSE(X_.dot(beta_ridge),z_)
		#These two lines are equivalent
		#z_predict_test[:,j] = model.fit(X_, z_).predict(X_test).ravel()
		#z_predict_train[:,j] = model.fit(X_, z_).predict(X_train).ravel()

	#print(np.shape(z_test))

	z_test_new = z_test[:,np.newaxis]
	z_train_new = z_train[:,np.newaxis]
	#print(np.shape(z_test_new))
	MSError_train[i] = np.mean(np.mean((z_train_new - z_predict_train)**2, axis=1, keepdims=True) )
	Bias_train[i] = np.mean( (z_train_new - np.mean(z_predict_train, axis=1, keepdims=True))**2 )
	Variance_train[i] = np.mean( np.var(z_predict_train, axis=1, keepdims=True) )
	MSError_test[i] = np.mean(np.mean((z_test_new - z_predict_test)**2, axis=1, keepdims=True) )
	Bias_test[i] = np.mean( (z_test_new - np.mean(z_predict_test, axis=1, keepdims=True))**2 )
	Variance_test[i] = np.mean( np.var(z_predict_test, axis=1, keepdims=True) )
	MSError_train_boot[i] = np.sum(MSE_boot)/NumBootstraps
	#print('Error:', MSError[i])
	#print('Bias^2:', Bias[i])
	#print('Var:', Variance[i])
	#print('{} >= {} + {} = {}'.format(MSError[i], Bias[i], Variance[i], Bias[i]+Variance[i]))

#model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))


#PlotErrors(ModelComplexity,MSError_train_boot,MSError_test,"MSE")
plt.plot(ModelComplexity,Variance_test,label="Test Variance")
plt.plot(ModelComplexity,Bias_test,label="Test Bias")
plt.plot(ModelComplexity,MSError_test,label="Test MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("Estimated error")
plt.legend()
plt.show()
