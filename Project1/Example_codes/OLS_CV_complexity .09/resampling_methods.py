# This program contains all resampling methods

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from statistical_functions import *
from data_processing import *
from print_and_plot import *
from regression_methods import *

seed = 2022

def Bootstrap(X, X_test, z_test, X_train, z_train, lamda, NumBootstraps, pol_deg, regression_method):

	z_predict_train = np.empty((len(z_train), NumBootstraps))
	z_predict_test = np.empty((len(z_test), NumBootstraps))
	MSE_boot = np.zeros(NumBootstraps)
	R2_curr = np.zeros(NumBootstraps)

	for j in range(NumBootstraps):

		X_, z_ = resample(X_train, z_train)

		if(regression_method == "Ridge"):
			beta = Ridge_beta(X_, z_, lamda, pol_deg)
			z_predict_test[:,j] = Ridge(X_, z_, X_test, lamda, pol_deg)
			MSE_boot[j] = MSE(Ridge(X_, z_, X_, lamda, pol_deg),z_)
		elif(regression_method == "LASSO"):
			beta = LASSO_SKL_beta(X_, z_,lamda)
			z_predict_test[:,j] = LASSO_SKL(X_, z_, X_test,lamda)
			MSE_boot[j] = MSE(LASSO_SKL(X_, z_, X_,lamda),z_)
		elif(regression_method == "OLS"):
			beta = OLS_beta(X_, z_)
			z_predict_test[:,j] = OLS(X_, z_, X_test)
			MSE_boot[j] = MSE(OLS(X_, z_, X_),z_)
		else:
			print("Don't forget to pick regression method!")

		R2_curr[j] = R2(z_test,z_predict_test[:,j])

	z_test_new = z_test[:,np.newaxis]

	MSError_test= np.mean(np.mean((z_test_new - z_predict_test)**2, axis=1, keepdims=True) )
	Bias_test = np.mean( (z_test_new - np.mean(z_predict_test, axis=1, keepdims=True))**2 )
	Variance_test = np.mean( np.var(z_predict_test, axis=1, keepdims=True) )
	MSError_train_boot = np.mean(MSE_boot)
	R2_test = np.mean(R2_curr)


	return MSError_test, Bias_test, Variance_test, MSError_train_boot, R2_test, beta

def Cross_Validation(nrk, x_scaled, y_scaled, z_scaled , lamda, pol_deg, regression_method):

	X = DesignMatrix(x_scaled,y_scaled,pol_deg)

	shuffled_indices = np.arange(len(x_scaled))
	np.random.shuffle(shuffled_indices)

	shuffled_matrix = np.zeros(X.shape)
	shuffled_z = np.zeros(len(x_scaled))

	for i in range(len(x_scaled)):
		shuffled_matrix[i] = X[shuffled_indices[i]]
		shuffled_z[i] = z_scaled[shuffled_indices[i]]

	split_matrix = np.split(shuffled_matrix,nrk)
	split_z = np.split(shuffled_z,nrk)

	MSError_test_CV= np.zeros(nrk)
	MSError_train_CV= np.zeros(nrk)
	R2_curr_test = np.zeros(nrk)
	R2_curr_train = np.zeros(nrk)
	z_plot_stored = np.zeros(len(z_scaled))

	for k in range(nrk):
		X_test = split_matrix[k]
		z_test = split_z[k]

		X_train = split_matrix
		X_train = np.delete(X_train,k,0)
		X_train = np.concatenate(X_train)

		z_train = split_z
		z_train = np.delete(z_train,k,0)
		z_train = np.ravel(z_train)

		if(regression_method == "Ridge"):
			beta = Ridge_beta(X_train, z_train, lamda, pol_deg)
			z_tilde = Ridge(X_train, z_train, X_train, lamda, pol_deg)
			z_predict = Ridge(X_train, z_train, X_test, lamda, pol_deg)
			z_plot = Ridge(X_train, z_train, X, lamda, pol_deg)
		elif(regression_method == "LASSO"):
			beta = LASSO_SKL_beta(X_train, z_train, lamda)
			z_tilde = LASSO_SKL(X_train, z_train, X_train, lamda)
			z_predict = LASSO_SKL(X_train, z_train, X_test, lamda)
			z_plot = LASSO_SKL(X_train, z_train, X, lamda)
		elif(regression_method == "OLS"):
			beta = OLS_beta(X_train, z_train)
			z_tilde = OLS(X_train, z_train, X_train)
			z_predict = OLS(X_train, z_train, X_test)
			z_plot = OLS(X_train, z_train, X)
		else:
			print("Don't forget to pick regression method!")

		z_plot_stored = z_plot_stored + z_plot
		R2_curr_test[k] = R2(z_test,z_predict)
		R2_curr_train[k] = R2(z_train,z_tilde)
		MSError_test_CV[k] = MSE(z_predict,z_test)
		MSError_train_CV[k] = MSE(z_tilde,z_train)

	return np.mean(MSError_test_CV), np.mean(MSError_train_CV), np.mean(R2_curr_test), np.mean(R2_curr_train), z_plot_stored/nrk


def NoResampling(x_scaled, y_scaled, z_scaled, pol_deg, lamb, regression_method):

	X = DesignMatrix(x_scaled,y_scaled,pol_deg)

	X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)

	if (regression_method == "OLS"):
		beta = OLS_beta(X_train, z_train)
		z_tilde = OLS(X_train, z_train, X_train)
		z_predict = OLS(X_train, z_train, X_test)
		z_plot = OLS(X_train, z_train, X)
	elif(regression_method == "Ridge"):
		beta = Ridge_beta(X_train, z_train, lamb, pol_deg)
		z_tilde = Ridge(X_train, z_train, X_train, lamb, pol_deg)
		z_predict = Ridge(X_train, z_train, X_test, lamb, pol_deg)
		z_plot = Ridge(X_train, z_train, X, lamb, pol_deg)
	elif(regression_method == "LASSO"):
		beta = LASSO_SKL_beta(X_train, z_train, lamb)
		z_tilde = LASSO_SKL(X_train, z_train, X_train, lamb)
		z_predict = LASSO_SKL(X_train, z_train, X_test, lamb)
		z_plot = LASSO_SKL(X_train, z_train, X, lamb)
	else:
		print("Don't forget to pick regression method!")

	R2_curr = R2(z_test,z_predict)
	MSError_test= MSE(z_predict,z_test)
	MSError_train = MSE(z_tilde,z_train)

	return MSError_test, MSError_train, R2_curr, z_plot




