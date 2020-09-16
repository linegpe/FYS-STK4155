from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed

from sklearn.model_selection import train_test_split
import scipy.linalg as scl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as skl

from statistical_functions import *
from data_processing import *
from print_and_plot import *
from regression_methods import *

#Polynomial degree, number of columns in the design matrix and number of data points
PolyDeg = 5
N = 20
regression_method ="OLS" # OLS, Ridge or LASSO
option = "none" #complexity or lambda

#Setting x and y arrays
x = np.arange(0, 1, 1.0/N)
y = np.arange(0, 1, 1.0/N)
#x = np.sort(np.random.uniform(0,1,N))
#y = np.sort(np.random.uniform(0,1,N))

x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Adding noise and raveling
#seed = 3489
seed = 3489
alpha = 0.01

z_noise = AddNoise(z,seed,alpha,N)
z_noise_flat = np.matrix.ravel(z_noise)
x_flat = np.ravel(x)
y_flat = np.ravel(y)

# Scaling data
x_scaled, y_scaled, z_scaled, scaler = DataScaling(x_flat,y_flat,z_noise_flat)

# Set up design matrix and scale the data
X = DesignMatrix(x_scaled,y_scaled,PolyDeg)
X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.36)

if (regression_method == "OLS"):
	beta = OLS_beta(X_train, z_train)
	z_tilde = OLS(X_train, z_train, X_train)
	z_predict = OLS(X_train, z_train, X_test)
	z_plot = OLS(X_train, z_train, X)
elif(regression_method == "Ridge"):
	beta = Ridge_beta(X_train, z_train, 0.01, PolyDeg)
	z_tilde = Ridge(X_train, z_train, X_train, 0.01, PolyDeg)
	z_predict = Ridge(X_train, z_train, X_test, 0.01, PolyDeg)
	z_plot = Ridge(X_train, z_train, X, 0.01, PolyDeg)
elif(regression_method == "LASSO"):
	beta = LASSO_SKL_beta(X_train, z_train, 0.01)
	z_tilde = LASSO_SKL(X_train, z_train, X_train, 0.01)
	z_predict = LASSO_SKL(X_train, z_train, X_test, 0.01)
	z_plot = LASSO_SKL(X_train, z_train, X, 0.01)

z_plot_split = np.zeros((N,N))
z_plot_split = np.reshape(z_plot, (N,N))
x_rescaled, y_rescaled, z_rescaled = DataRescaling(x_scaled,y_scaled,z_plot,scaler,N)

# Print error estimations and plot the surfaces
#PrintErrors(z_train,z_tilde,z_test,z_predict)
#SurfacePlot(x,y,z_noise,z_rescaled)

#-------------------------------------------------------Variance-bias tradeoff----------------------------------------------------------

NumBootstraps = 100

# Complexity study
MaxPolyDeg = 25
ModelComplexity =np.arange(0,MaxPolyDeg+1)

if(option == "complexity"):

	MSError_test= np.zeros(MaxPolyDeg+1)
	Bias_test =np.zeros(MaxPolyDeg+1)
	Variance_test =np.zeros(MaxPolyDeg+1)

	MSError_train_boot = np.zeros(MaxPolyDeg+1)

	for i in range(MaxPolyDeg+1):
		X_new = DesignMatrix(x_scaled,y_scaled,i)
		X_train, X_test, z_train, z_test = train_test_split(X_new, z_scaled, test_size=0.2)

		z_predict_train = np.empty((len(z_train), NumBootstraps))
		z_predict_test = np.empty((len(z_test), NumBootstraps))
		MSE_boot = np.zeros(NumBootstraps)

		for j in range(NumBootstraps):
			X_, z_ = resample(X_train, z_train)

			if(regression_method == "Ridge"):
				beta = Ridge_beta(X_, z_, 0.01, i)
				z_predict_test[:,j] = Ridge(X_, z_, X_test, 0.01, i)
				MSE_boot[j] = MSE(Ridge(X_, z_, X_, 0.01, i),z_)
			elif(regression_method == "LASSO"):
				beta = LASSO_SKL_beta(X_, z_,0.01)
				z_predict_test[:,j] = LASSO_SKL(X_, z_, X_test,0.01)
				MSE_boot[j] = MSE(LASSO_SKL(X_, z_, X_,0.01),z_)
			elif(regression_method == "OLS"):
				beta = OLS_beta(X_, z_)
				z_predict_test[:,j] = OLS(X_, z_, X_test)
				MSE_boot[j] = MSE(OLS(X_, z_, X_),z_)

		z_test_new = z_test[:,np.newaxis]

		MSError_test[i] = np.mean(np.mean((z_test_new - z_predict_test)**2, axis=1, keepdims=True) )
		Bias_test[i] = np.mean( (z_test_new - np.mean(z_predict_test, axis=1, keepdims=True))**2 )
		Variance_test[i] = np.mean( np.var(z_predict_test, axis=1, keepdims=True) )
		MSError_train_boot[i] = np.sum(MSE_boot)/NumBootstraps


	PlotErrors(ModelComplexity,MSError_train_boot,MSError_test,"MSE")
	plt.plot(ModelComplexity,Variance_test,label="Test Variance")
	plt.plot(ModelComplexity,Bias_test,label="Test Bias")
	plt.plot(ModelComplexity,MSError_test,label="Test MSE")
	plt.xlabel("Polynomial degree")
	plt.ylabel("Estimated error")
	plt.legend()
	plt.show()

nlambdas = 11
lambdas = np.logspace(-5, 5, nlambdas)
if(option == "lambda"):

	MSError_test= np.zeros(nlambdas)
	Bias_test =np.zeros(nlambdas)
	Variance_test =np.zeros(nlambdas)

	MSError_train_boot = np.zeros(nlambdas)
	betas = np.empty((nlambdas,int((PolyDeg+2)*(PolyDeg+1)/2)))

	X = DesignMatrix(x_scaled,y_scaled,PolyDeg)
	X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)

	for lam in range(nlambdas):
		z_predict_train = np.empty((len(z_train), NumBootstraps))
		z_predict_test = np.empty((len(z_test), NumBootstraps))
		MSE_boot = np.zeros(NumBootstraps)

		for j in range(NumBootstraps):
			X_, z_ = resample(X_train, z_train)

			if(regression_method == "Ridge"):
				beta = Ridge_beta(X_, z_, lambdas[lam], PolyDeg)
				z_predict_test[:,j] = Ridge(X_, z_, X_test, lambdas[lam], PolyDeg)
				MSE_boot[j] = MSE(Ridge(X_, z_, X_, lambdas[lam], PolyDeg),z_)
			elif(regression_method == "LASSO"):
				beta = LASSO_SKL_beta(X_, z_, lambdas[lam])
				z_predict_test[:,j] = LASSO_SKL(X_, z_, X_test, lambdas[lam])
				MSE_boot[j] = MSE(LASSO_SKL(X_, z_, X_, lambdas[lam]),z_)
			betas[lam,:]=beta

		z_test_new = z_test[:,np.newaxis]
		MSError_test[lam] = np.mean(np.mean((z_test_new - z_predict_test)**2, axis=1, keepdims=True) )
		Bias_test[lam] = np.mean( (z_test_new - np.mean(z_predict_test, axis=1, keepdims=True))**2 )
		Variance_test[lam] = np.mean( np.var(z_predict_test, axis=1, keepdims=True) )
		MSError_train_boot[lam] = np.sum(MSE_boot)/NumBootstraps

	plt.plot(lambdas,MSError_test,label=" test")
	plt.plot(lambdas,MSError_train_boot,label=" train")
	plt.xlabel("lambda")
	plt.ylabel("Estimated error")
	plt.xscale("log")
	plt.legend()
	plt.show()

	for i in range(int((PolyDeg+2)*(PolyDeg+1)/2)):
		plt.plot(lambdas,betas[:,i],"r")
	plt.xlabel("lambda")
	plt.ylabel("beta number")
	plt.xscale("log")
	plt.show()


	# -------------------- Cross validation -------------------
nrk = 5
MSError_test_CV_full = np.zeros(MaxPolyDeg+1)
for m in range(MaxPolyDeg+1):

	shuffled_indices = np.arange(len(x_scaled))
	np.random.shuffle(shuffled_indices)


	X = DesignMatrix(x_scaled,y_scaled,m)

	shuffled_matrix = np.zeros(X.shape)
	shuffled_z = np.zeros(len(x_scaled))
	for i in range(len(x_scaled)):
		shuffled_matrix[i] = X[shuffled_indices[i]]
		shuffled_z[i] = z_scaled[shuffled_indices[i]]

	split_matrix = np.split(shuffled_matrix,nrk)
	split_z = np.split(shuffled_z,nrk)

	MSError_test_CV= np.zeros(nrk)

	for k in range(nrk):
		X_test = split_matrix[k]
		z_test = split_z[k]

		X_train = split_matrix
		X_train = np.delete(X_train,k,0)
		X_train = np.concatenate(X_train)

		z_train = split_z
		z_train = np.delete(z_train,k,0)
		z_train = np.ravel(z_train)

		if (regression_method == "OLS"):
			beta = OLS_beta(X_train, z_train)
			z_tilde = OLS(X_train, z_train, X_train)
			z_predict = OLS(X_train, z_train, X_test)
			z_plot = OLS(X_train, z_train, X)
		elif(regression_method == "Ridge"):
			beta = Ridge_beta(X_train, z_train, 0.01, m)
			z_tilde = Ridge(X_train, z_train, X_train, 0.01, m)
			z_predict = Ridge(X_train, z_train, X_test, 0.01, m)
			z_plot = Ridge(X_train, z_train, X, 0.01, m)
		elif(regression_method == "LASSO"):
			beta = LASSO_SKL_beta(X_train, z_train, 0.01)
			z_tilde = LASSO_SKL(X_train, z_train, X_train, 0.01)
			z_predict = LASSO_SKL(X_train, z_train, X_test, 0.01)
			z_plot = LASSO_SKL(X_train, z_train, X, 0.01)


		MSError_test_CV[k] = MSE(z_predict,z_test)

	MSError_test_CV_full[m] = np.mean(MSError_test_CV)

plt.plot(ModelComplexity,MSError_test_CV_full)
plt.show()

	#Bias_test =np.zeros(nlambdas)
	#Variance_test =np.zeros(nlambdas)
