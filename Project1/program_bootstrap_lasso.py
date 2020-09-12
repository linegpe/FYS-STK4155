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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.linear_model as skl

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
MaxPolyDeg = 13

nlambdas = 100
lambdas = np.logspace(-3, 5, nlambdas)

#ModelComplexity =np.arange(0,MaxPolyDeg+1)

MSError_train = np.zeros(nlambdas)
Bias_train =np.zeros(nlambdas)
Variance_train =np.zeros(nlambdas)

MSError_test= np.zeros(nlambdas)
Bias_test =np.zeros(nlambdas)
Variance_test =np.zeros(nlambdas)

MSError_train_boot = np.zeros(nlambdas)


#X = DesignMatrix(x_scaled,y_scaled,PolyDeg)
X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)
I = np.identity(int((PolyDeg+2)*(PolyDeg+1)/2))
X = DesignMatrix(x_scaled,y_scaled,PolyDeg)

for lam in range(nlambdas):
	z_predict_train = np.empty((len(z_train), NumBootstraps))
	z_predict_test = np.empty((len(z_test), NumBootstraps))
	MSE_boot = np.zeros(NumBootstraps)

	#ridge = Ridge(alpha = lam)

	for j in range(NumBootstraps):
		X_, z_ = resample(X_train, z_train)
		#beta_ridge = np.linalg.pinv(X_.T.dot(X_)+lambdas[lam]*I).dot(X_.T.dot(z_))
		lasso_fit = skl.Lasso(alpha=lambdas[lam]).fit(X_,z_)

		z_predict_test[:,j] = lasso_fit.predict(X_test)
		z_predict_train = lasso_fit.predict(X_)
		MSE_boot[j] = MSE(z_predict_train,z_)
		"""ridge.fit(X_, z_)
		z_predict_train[:,j] = ridge.predict(X_train)
		z_predict_test[:,j] = ridge.predict(X_test)
		MSE_boot[j] = MSE(ridge.predict(X_),z_)"""

	z_test_new = z_test[:,np.newaxis]
	z_train_new = z_train[:,np.newaxis]
	MSError_test[lam] = np.mean(np.mean((z_test_new - z_predict_test)**2, axis=1, keepdims=True) )
	Bias_test[lam] = np.mean( (z_test_new - np.mean(z_predict_test, axis=1, keepdims=True))**2 )
	Variance_test[lam] = np.mean( np.var(z_predict_test, axis=1, keepdims=True) )
	MSError_train_boot[lam] = np.sum(MSE_boot)/NumBootstraps

#PlotErrors(lambdas,MSError_train_boot,MSError_test,"MSE")

plt.plot(ModelComplexity,Variance_test,label="Test Variance")
plt.plot(ModelComplexity,Bias_test,label="Test Bias")
plt.plot(ModelComplexity,MSError_test,label="Test MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("Estimated error")
plt.legend()
plt.show()​

"""
plt.plot(lambdas,MSError_test,label=" test")
plt.plot(lambdas,MSError_train_boot,label=" train")
plt.xlabel("lambda")
plt.ylabel("Estimated error")
plt.xscale("log")
    #plt.ylim(-0.1,1.2)
plt.legend()
plt.show()
"""
