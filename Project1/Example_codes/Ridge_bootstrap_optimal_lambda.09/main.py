# The main program is used to collecting results for the Franke's function

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from imageio import imread

from sklearn.model_selection import train_test_split
import scipy.linalg as scl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from statistical_functions import *
from data_processing import *
from print_and_plot import *
from regression_methods import *
from resampling_methods import *

# Setting polynomial degree, number of columns in the design matrix and number of data points
PolyDeg = 5
N = 20

# Choosing options
regression_method ="Ridge" # "OLS", "Ridge" or "LASSO"
option = "lambda" #"complexity" or "lambda"
resampling = "bootstrap" # "bootstrap", "cv" or "no resampling"

# Setting x and y arrays, setting Frankie's function
x = np.arange(0, 1, 1.0/N)
y = np.arange(0, 1, 1.0/N)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

# Adding noise and raveling
seed = 2022
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

# Simple regression in case of 1 polynomial degree (done for testing different options before applying resampling)
if (regression_method == "OLS"):
	beta = OLS_beta(X_train, z_train)
	z_tilde = OLS(X_train, z_train, X_train)
	z_predict = OLS(X_train, z_train, X_test)
	z_plot = OLS(X_train, z_train, X)
elif(regression_method == "Ridge"):
	lamb = 0.01 
	beta = Ridge_beta(X_train, z_train, lamb, PolyDeg)
	z_tilde = Ridge(X_train, z_train, X_train, lamb, PolyDeg)
	z_predict = Ridge(X_train, z_train, X_test, lamb, PolyDeg)
	z_plot = Ridge(X_train, z_train, X, lamb, PolyDeg)
elif(regression_method == "LASSO"):
	lamb = 0.01
	beta = LASSO_SKL_beta(X_train, z_train, lamb)
	z_tilde = LASSO_SKL(X_train, z_train, X_train, lamb)
	z_predict = LASSO_SKL(X_train, z_train, X_test, lamb)
	z_plot = LASSO_SKL(X_train, z_train, X, lamb)

# Rescaling data
x_rescaled, y_rescaled, z_rescaled = DataRescaling(x_scaled,y_scaled,z_plot,scaler,N,N)

# Print error estimations and plot the surfaces
#PrintErrors(z_train,z_tilde,z_test,z_predict)
#Plot_Franke_Test_Train(z_test,z_train, X_test, X_train, scaler, x, y, z_noise)
#Plot_Beta_with_Err(beta, X, alpha, 1)

#-------------------------------------------------------Variance-bias tradeoff----------------------------------------------------------

NumBootstraps = 100

# Complexity study
MaxPolyDeg = 5
lamb = 0.01
ModelComplexity =np.arange(0,MaxPolyDeg+1)

# lambda study
nlambdas = 100
lambdas = np.logspace(-5, 5, nlambdas)

# Preparing the file
filename = "Files/d_MSE_test50.txt"
f = open(filename, "w")
f.write("MSE test   MSE train\n")
f.close()

if(resampling == "bootstrap"):

	# -------------------- Bootstrap -------------------

	if(option == "complexity"):

		f = open(filename, "a")
		MSError_test= np.zeros(MaxPolyDeg+1)
		Bias_test =np.zeros(MaxPolyDeg+1)
		Variance_test =np.zeros(MaxPolyDeg+1)
		R2_test =np.zeros(MaxPolyDeg+1)
		MSError_train_boot = np.zeros(MaxPolyDeg+1)

		for i in range(MaxPolyDeg+1):

			X = DesignMatrix(x_scaled,y_scaled,i)
			X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)
			MSError_test[i], Bias_test[i], Variance_test[i], MSError_train_boot[i], R2_test[i],beta = Bootstrap(X, X_test, z_test, X_train, z_train , lamb, NumBootstraps, i, regression_method)
			f.write("{0} {1}\n".format(MSError_test[i], MSError_train_boot[i]))

		f.close()
		Plot_MSE_Test_Train(ModelComplexity, MSError_test, MSError_train_boot, regression_method, resampling, "model complexity",option)
		Plot_Bias_Variance(ModelComplexity, MSError_test, Variance_test, Bias_test, regression_method, resampling, option)

	if(option == "lambda"):

		f = open(filename, "a")
		MSError_test= np.zeros(nlambdas)
		Bias_test =np.zeros(nlambdas)
		Variance_test =np.zeros(nlambdas)
		R2_test =np.zeros(nlambdas)
		MSError_train_boot = np.zeros(nlambdas)
		betas = np.empty((nlambdas,int((PolyDeg+2)*(PolyDeg+1)/2)))

		X = DesignMatrix(x_scaled,y_scaled,PolyDeg)
		X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)

		for lam in range(nlambdas):
			MSError_test[lam], Bias_test[lam], Variance_test[lam], MSError_train_boot[lam], R2_test[lam], betas[lam,:] = Bootstrap(X, X_test, z_test, X_train, z_train, lambdas[lam], NumBootstraps, PolyDeg, regression_method)
			f.write("{0} {1}\n".format(MSError_test[lam], MSError_train_boot[lam]))

		f.close()
		Plot_MSE_Test_Train(lambdas, MSError_test, MSError_train_boot, regression_method, resampling, "$\lambda$",option)
		Plot_Bias_Variance(lambdas, MSError_test, Variance_test, Bias_test, regression_method, resampling, option)
		Plot_Betas(PolyDeg, betas, lambdas, regression_method, resampling)

        # Minimum element for a given polynomial degree
		MinElement(MSError_test, R2_test, lambdas)

elif(resampling == "cv"):

	# -------------------- Cross validation -------------------
	nrk = 5

	if(option == "complexity"):

		f = open(filename, "a")

		MSError_test = np.zeros(MaxPolyDeg+1)
		MSError_train = np.zeros(MaxPolyDeg+1)
		R2_test = np.zeros(MaxPolyDeg+1)
		R2_train = np.zeros(MaxPolyDeg+1)

		for i in range(MaxPolyDeg+1):
			MSError_test[i], MSError_train[i], R2_test[i], R2_train[i], z_pred = Cross_Validation(nrk, x_scaled, y_scaled, z_scaled, lamb, i, regression_method)
			f.write("{0} {1}\n".format(MSError_test[i], MSError_train[i]))

		f.close()
		Plot_MSE_Test_Train(ModelComplexity, MSError_test, MSError_train, regression_method, resampling, "model complexity",option)

	elif(option == "lambda"):

		f = open(filename, "a")

		MSError_test = np.zeros(nlambdas)
		MSError_train = np.zeros(nlambdas)
		R2_test =np.zeros(nlambdas)
		R2_train =np.zeros(nlambdas)

		for lam in range(nlambdas):
			MSError_test[lam], MSError_train[lam], R2_test[lam], R2_train[lam], z_pred = Cross_Validation(nrk, x_scaled, y_scaled, z_scaled, lambdas[lam], PolyDeg, regression_method)
			f.write("{0} {1}\n".format(MSError_test[lam], MSError_train[lam]))

		f.close()
		Plot_MSE_Test_Train(lambdas, MSError_test, MSError_train, regression_method, resampling, "$\lambda$", option)

        # Minimum element for a given polynomial degree
		MinElement(MSError_test, R2_test, lambdas)


elif(resampling == "no resampling"):
	# -------------------- No resampling -------------------

	if(option == "complexity"):

		f = open(filename, "a")

		MSError_test = np.zeros(MaxPolyDeg+1)
		MSError_train = np.zeros(MaxPolyDeg+1)
		R2_test = np.zeros(MaxPolyDeg+1)

		for i in range(MaxPolyDeg+1):	
			MSError_test[i], MSError_train[i], R2_test[i], z_pred = NoResampling(x_scaled, y_scaled, z_scaled, i, lamb, regression_method)
			f.write("{0} {1}\n".format(MSError_test[i], MSError_train[i]))	

		f.close()
		Plot_MSE_Test_Train(ModelComplexity, MSError_test, MSError_train, regression_method, resampling, "model complexity",option)

	elif(option == "lambda"):

		f = open(filename, "a")

		MSError_test= np.zeros(nlambdas)
		MSError_train= np.zeros(nlambdas)
		R2_test =np.zeros(nlambdas)

		for lam in range(nlambdas):
			MSError_test[lam], MSError_train[lam], R2_test[lam], z_pred = NoResampling(x_scaled, y_scaled, z_scaled, PolyDeg, lambdas[lam], regression_method)
			f.write("{0} {1}\n".format(MSError_test[lam], MSError_train[lam]))

		f.close()
		Plot_MSE_Test_Train(lambdas, MSError_test, MSError_train, regression_method, resampling, "$\lambda$", option)

        # Minimum element for a given polynomial degree
		MinElement(MSError_test, R2_test, lambdas)

else:

	print("Pick resampling option next time!")
