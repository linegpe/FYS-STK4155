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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

fig = plt.figure()
ax = fig.gca(projection='3d')

#Setting x and y arrays
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

#Polynomial degree, number of columns in the design matrix and number of data points
PolyDeg = 5
N = 20


# Making a default Design Matrix

def DesignMatrix(x, y, PolyDeg):
	x = np.ravel(x)
	y = np.ravel(y)

	NumElem = int((PolyDeg+2)*(PolyDeg+1)/2)
	X = np.ones((len(x),NumElem))

	for i in range(1,PolyDeg+1):
		j = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,j+k] = x**(i-k)*y**k
	return(X)


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def R2(z_data, z_model):
	return 1 - np.sum((z_data - z_model) ** 2)/np.sum((z_data - np.mean(z_data)) ** 2)

def MSE(z_data,z_model):
	n = np.size(z_model)
	return np.sum((z_data-z_model)**2)/n

#SVD matrix inversion beta
def ols_svd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y

def BetaVar(X, alpha):
    return  np.diag(alpha**2*np.linalg.pinv(X.T.dot(X)))

x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)

#Adding noise
seed = 2022
#2022 is great!

np.random.seed(seed)
alpha = 0.01
noise = alpha*np.random.randn(N, N)
z_noise_orig = z+noise
z_noise = np.matrix.ravel(z_noise_orig)

x_flat = np.ravel(x)
y_flat = np.ravel(y)
print(np.shape(z_noise))
dataset = np.stack((x_flat, y_flat, z_noise)).T
scaler = StandardScaler()
scaler.fit(dataset)
[x_scaled, y_scaled, z_scaled] = scaler.transform(dataset).T

X = DesignMatrix(x_scaled,y_scaled,PolyDeg)


#print(x)
#print(x_scaled)

X_train, X_test, z_train, z_test = train_test_split(X, z_scaled, test_size=0.2)

#Beta from the OLS
beta = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(z_train))
#beta = ols_svd(X_train,z_train)

#print(beta)

z_tilde = X_train.dot(beta)
z_predict = X_test.dot(beta)
z_plot = X.dot(beta)
z_plot_split = np.zeros((N,N))
k = 0
for i in range(N):
	for j in range(N):
		z_plot_split[i,j]=z_plot[j+k]
	k+=N


dataset_scaled = np.stack((x_scaled, y_scaled, z_plot))
dataset_rescaled = scaler.inverse_transform(dataset_scaled.T)
print(dataset_rescaled[0])

z_rescaled = np.zeros((N,N))
k = 0
for i in range(N):
	for j in range(N):
		z_rescaled[i,j]=dataset_rescaled[j+k,2]
	k+=N



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
MaxPolyDeg = 13

ModelComplexity =np.arange(0,MaxPolyDeg+1)

MSError= np.zeros(MaxPolyDeg+1)
Bias =np.zeros(MaxPolyDeg+1)
Variance =np.zeros(MaxPolyDeg+1)

#z_noise = np.ravel(z_noise)

#for i in range(MaxPolyDeg+1):
#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)

for i in range(MaxPolyDeg+1):

	X_new = DesignMatrix(x_scaled,y_scaled,i)
	#z_noise_pred = np.empty((len(z_noise[0]), NumBootstraps))
	model = LinearRegression(fit_intercept=False)

	X_train, X_test, z_train, z_test = train_test_split(X_new, z_scaled, test_size=0.2)

	z_predict = np.empty((len(z_test), NumBootstraps))
	#z_test_new = z_test[:,np.newaxis]

	for j in range(NumBootstraps):
		X_, z_ = resample(X_train, z_train)
		z_predict[:,j] = model.fit(X_, z_).predict(X_test).ravel()
    
	#print(np.shape(z_test))
	z_test_new = z_test[:,np.newaxis]
	#print(np.shape(z_test_new))
	MSError[i] = np.mean(np.mean((z_test_new - z_predict)**2, axis=1, keepdims=True) )
	Bias[i] = np.mean( (z_test_new - np.mean(z_predict, axis=1, keepdims=True))**2 )
	Variance[i] = np.mean( np.var(z_predict, axis=1, keepdims=True) )
	#print('Error:', MSError[i])
	#print('Bias^2:', Bias[i])
	#print('Var:', Variance[i])
	#print('{} >= {} + {} = {}'.format(MSError[i], Bias[i], Variance[i], Bias[i]+Variance[i]))

#model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

plt.plot(ModelComplexity,Bias, label='Bias')
plt.plot(ModelComplexity,Variance, label='Variance')
plt.plot(ModelComplexity,MSError, label='MSE')
plt.legend()
plt.show()
