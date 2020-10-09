# The main program is used to collecting results for the terrain data

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from random import random, seed
import random

from data_processing import *
from resampling_methods import *


# Load the terrain

N, M, x_mesh, y_mesh, z_mesh, terrain, x_flat, y_flat, z_flat = SetTerrainCompression(50)

# Plot the original surface.
PlotTerrain(terrain, y_mesh, x_mesh, z_mesh)

# Scaling data
x_scaled, y_scaled, z_scaled, scaler = DataScaling(x_flat,y_flat,z_flat)

# --------------------------- Complexity study with cross-validation -------------------------

# Preparing the files
filename_z = "Files/file.txt"
f_z = open(filename_z, "w")
f_z.write("Predicted value\n")
f_z.close()

filename_r = "Files/another_file.txt"
f_r = open(filename_r, "w")
f_r.write("R2 score\n")
f_r.close()

filename_r_test = "Files/file_test.txt"
f_r_test = open(filename_r_test, "w")
f_r_test.write("R2 test score\n")
f_r_test.close()

filename_r_train = "Files/file_train.txt"
f_r_train = open(filename_r_train, "w")
f_r_train.write("R2 train score\n")
f_r_train.close()

MaxPolyDeg = 2
ModelComplexity =np.arange(0,MaxPolyDeg+1)
seed = 2022
np.random.seed(seed)

PolyDeg = 10
nlambdas = 10
lambdas = np.logspace(-3, 5, nlambdas)

regression_method ="OLS" # "OLS", "Ridge" or "LASSO"
resampling = "cv" # "cv" or "no resampling"
option = "complexity" #"complexity" or "lambda"
search = "on" # Search of the optimal lambda, "on" or "off"
gridsearch = "off" # Search of the optimal lambda, "on" or "off"

nrk = 8

MSError_test= np.zeros(MaxPolyDeg+1)
MSError_train = np.zeros(MaxPolyDeg+1)
R2_test = np.zeros(MaxPolyDeg+1)
R2_train = np.zeros(MaxPolyDeg+1)
lamb = 0.01

f_z = open(filename_z, "a")
f_r = open(filename_r, "a")
f_r_test = open(filename_r_test, "a")
f_r_train = open(filename_r_train, "a")

if(resampling == "cv"):
	# --------------------------- Complexity study with cross-validation -------------------------

	for i in range(MaxPolyDeg+1):
	
		MSError_test[i], MSError_train[i], R2_test[i], R2_train[i], z_pred = Cross_Validation(nrk, x_scaled, y_scaled, z_scaled, lamb, i, regression_method)
		print(i)
		f_r.write("{0}\n".format(R2_test[i]))

	for i in range(len(z_pred)):
		f_z.write("{0}\n".format(z_pred[i]))
	f_z.close()
	f_r.close()
	print(np.argmin(MSError_test))

elif(resampling == "no resampling"):

	# --------------------------- Complexity study without cross-validation -------------------------

	for i in range(MaxPolyDeg+1):
	
		MSError_test[i], MSError_train[i], R2_test[i], z_plot = NoResampling(x_scaled, y_scaled, z_scaled, i, lamb, regression_method)
		f_r.write("{0}\n".format(R2_test[i]))
	
	for i in range(len(z_plot)):
		f_z.write("{0}\n".format(z_plot[i]))
	f_z.close()
	f_r.close()
	
Plot_MSE_Test_Train(ModelComplexity, MSError_test, MSError_train, regression_method, resampling, "model complexity",option)
PlotSurfacesTerrain(x_scaled, y_scaled, scaler, N, M)
#Plot_R2_Terrain(ModelComplexity, regression_method)

if(search == "on"):

	#-----------------------Search for the best lambda -------------------------

	MSError_test= np.zeros(nlambdas)
	MSError_train = np.zeros(nlambdas)
	R2_test =np.zeros(nlambdas)
	R2_train =np.zeros(nlambdas)
	
	for lam in range(nlambdas):
		if(resampling=="cv"):
			MSError_test[lam], MSError_train[lam], R2_test[lam], R2_train[lam], z_pred = Cross_Validation(nrk, x_scaled, y_scaled, z_scaled, lambdas[lam], PolyDeg, regression_method)
		elif(resampling=="no resampling"):
			MSError_test[lam], MSError_train[lam], R2_test[lam], z_plot = NoResampling(x_scaled, y_scaled, z_scaled, PolyDeg, lambdas[lam], regression_method)
		f_r_test.write("{0}\n".format(R2_test[lam]))
		f_r_train.write("{0}\n".format(R2_train[lam]))
		print(lam)

	f_r_test.close()
	f_r_train.close()

	MinElement(MSError_test, R2_test, lambdas)
	Plot_R2_all_Terrain(lambdas)

if(gridsearch == "on"):

	#-----------------------Search for the best lambda -------------------------

	#redefinition of the functions 
	MaxPolyDeg = 25
	ModelComplexity =np.arange(1,MaxPolyDeg+1)
	nlambdas = 300
	lambdas = np.logspace(-7, 3, nlambdas)
	MSError_test= np.zeros((MaxPolyDeg, nlambdas))
	MSError_train = np.zeros((MaxPolyDeg, nlambdas))
	R2_test = np.zeros((MaxPolyDeg, nlambdas))
	R2_train = np.zeros((MaxPolyDeg, nlambdas))
	min_el_def= 10000
	lambd_min = 0
	p_min = 0
	lam_min = 0
	
	for i in range(1,MaxPolyDeg+1):
		for lam in range(nlambdas):
			MSError_test[i-1,lam], MSError_train[i-1,lam], R2_test[i-1,lam], R2_train[i-1,lam], z_pred = Cross_Validation(nrk, x_scaled, y_scaled, z_scaled, lambdas[lam], i, regression_method)
		min_el_curr = np.min(MSError_test[i-1,:])
		if(min_el_curr<min_el_def):
			min_el_def = min_el_curr
			lambd_min = lambdas[np.argmin(MSError_test[i-1,:])]
			p_min = i
			lam_min = lam
		print(i)

	print("Minimal MSE: ",min_el_def)
	print("Minimal lambda: ",lambd_min)
	print("Minimal polynomial degree: ",p_min)
	print("Minimal R2: ", R2_test[p_min-1, lam])

	print(np.shape(MSError_test))
	x_mesh, y_mesh = np.meshgrid(lambdas,ModelComplexity) 
	print(np.shape(y_mesh))
	SurfacePlot(y_mesh,x_mesh, MSError_test, MSError_test)
	plt.show()


# ----------------------- Plotting the surface for optimal parameters ----------------

PolyDeg = 18
lambd = 0.000149926843278604 

MSError_test, MSError_train, R2_test, R2_train, z_pred = Cross_Validation(nrk, x_scaled, y_scaled, z_scaled, lambd, PolyDeg, regression_method)
print(MSError_test)
# Rescaling data
x_rescaled, y_rescaled, z_rescaled = DataRescaling(x_scaled,y_scaled,z_pred,scaler,N,M)

SurfacePlot(y_rescaled,x_rescaled,z_rescaled,z_rescaled)

# ----------------------- Studying beta in the simplest case of polynomial degree 5, OLS, no crodss-validation ----------------

PolyDeg = 5
# Set up design matrix and scale the data
X = DesignMatrix(x_scaled,y_scaled,PolyDeg)

X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)

if (regression_method == "OLS"):
	beta = OLS_beta(X_train, z_train)
	z_tilde = OLS(X_train, z_train, X_train)
	z_predict = OLS(X_train, z_train, X_test)
	z_plot = OLS(X_train, z_train, X)

	N = len(z_plot)
	p=(PolyDeg+1)*(PolyDeg+2)/2
	sigma_sq = 1/(N-p-1)*np.sum((z_scaled-z_plot)**2)
	
	Plot_Beta_with_Err(beta, X, 1, sigma_sq)