from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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

fig = plt.figure()
ax = fig.gca(projection='3d')

#Setting x and y arrays
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

#Polynomial degree, number of columns in the design matrix and number of data points
PolyDeg = 5
N = 20

x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Adding noise
#seed = 2022
seed = 2000
np.random.seed(seed)
alpha = 0.01
noise = alpha*np.random.randn(N, N)
z_noise_orig = z+noise
z_noise = np.matrix.ravel(z_noise_orig)


x_flat = np.ravel(x)
y_flat = np.ravel(y)

# Scaling data
x_scaled, y_scaled, z_scaled, scaler = DataScaling(x_flat,y_flat,z_noise)

X = DesignMatrix(x_scaled,y_scaled,PolyDeg)
X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)

#Beta from the OLS
beta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(z_train))
#beta = ols_svd(X_train,z_train)

z_tilde = X_train.dot(beta)
z_predict = X_test.dot(beta)
z_plot = X.dot(beta)
z_plot_split = np.zeros((N,N))
k = 0

z_plot_split = np.reshape(z_plot, (N,N))
x_rescaled, y_rescaled, z_rescaled = DataRescaling(x_scaled,y_scaled,z_plot,scaler,N)


print("Training R2")
print(R2(z_train,z_tilde))

print("Training MSE")
print(MSE(z_train,z_tilde))

print("Test R2")
print(R2(z_test,z_predict))

print("Test MSE")
print(MSE(z_test,z_predict))

print("Variance for beta")
#print(BetaVar(X,alpha))

# Plot the surface.
surf = ax.plot_surface(x, y, z_noise_orig, cmap=cm.ocean,
linewidth=0, antialiased=False)


surf = ax.plot_surface(x, y, z_rescaled, cmap=cm.Pastel1,
linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()




#Splitting of the data into training and test sets


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

for i in range(MaxPolyDeg+1):
	X_new = DesignMatrix(x_scaled,y_scaled,i)
	#z_noise_pred = np.empty((len(z_noise[0]), NumBootstraps))
	model = LinearRegression(fit_intercept=False)

	X_train, X_test, z_train, z_test = train_test_split(X_new, z_scaled, test_size=0.2)

	z_predict_train = np.empty((len(z_train), NumBootstraps))
	z_predict_test = np.empty((len(z_test), NumBootstraps))
	#z_test_new = z_test[:,np.newaxis]

	for j in range(NumBootstraps):
		X_, z_ = resample(X_train, z_train)
		z_predict_test[:,j] = model.fit(X_, z_).predict(X_test).ravel()
		z_predict_train[:,j] = model.fit(X_, z_).predict(X_train).ravel()

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
	#print('Error:', MSError[i])
	#print('Bias^2:', Bias[i])
	#print('Var:', Variance[i])
	#print('{} >= {} + {} = {}'.format(MSError[i], Bias[i], Variance[i], Bias[i]+Variance[i]))

#model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

#plt.plot(ModelComplexity,Bias_train, label='Bias train ')
#plt.plot(ModelComplexity,Variance_train, label='Variance train ')
plt.plot(ModelComplexity,MSError_train, label='MSE train ')
#plt.plot(ModelComplexity,Bias_test, label='Bias test')
#plt.plot(ModelComplexity,Variance_test, label='Variance test')
plt.plot(ModelComplexity,MSError_test, label='MSE test')
plt.legend()
plt.show()
