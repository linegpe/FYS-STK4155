import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.linear_model as skl
from statistical_functions import *

def OLS(X_train, z_train, X_test):
	beta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(z_train))
	z_predict = X_test.dot(beta)
	return z_predict

def OLS_beta(X_train, z_train):
	return np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(z_train))

def OLS_SVD(X_train, z_train, X_test):
	beta = ols_svd(X_train, z_train)
	z_predict = X_test.dot(beta)
	return z_predict

def OLS_SVD_beta(X_train, z_train):
	return ols_svd(X_train, z_train)

def Ridge(X_train, z_train, X_test, lamb, pol_deg):
	I = np.identity(int((pol_deg+2)*(pol_deg+1)/2))
	beta = np.linalg.pinv(X_train.T.dot(X_train)+lamb*I).dot(X_train.T.dot(z_train))
	z_predict = X_test.dot(beta)
	return z_predict

def Ridge_beta(X_train, z_train, lamb, pol_deg):
	I = np.identity(int((pol_deg+2)*(pol_deg+1)/2))
	return np.linalg.pinv(X_train.T.dot(X_train)+lamb*I).dot(X_train.T.dot(z_train))


#def Ridge_SKL_beta():


def LASSO_SKL(X_train, z_train, X_test, lamb):
	lasso_fit = skl.Lasso(alpha=lamb, max_iter=10e5, tol=0.01, normalize=True, fit_intercept=False).fit(X_train,z_train)
	z_predict = lasso_fit.predict(X_test)
	beta = lasso_fit.coef_
	return z_predict

def LASSO_SKL_beta(X_train, z_train, lamb):
	lasso_fit = skl.Lasso(alpha=lamb, max_iter=10e5, tol=0.01, normalize=True, fit_intercept=False).fit(X_train,z_train)
	return lasso_fit.coef_

