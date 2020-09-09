import numpy as np
from sklearn.preprocessing import StandardScaler


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


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


def DataScaling(x,y,z):
	# x and y must be raveled
	dataset = np.stack((x, y, z)).T
	scaler = StandardScaler()
	scaler.fit(dataset)
	[x_scaled, y_scaled, z_scaled] = scaler.transform(dataset).T
	return x_scaled, y_scaled, z_scaled, scaler

def DataRescaling(x,y,z,scaler,N):
	dataset_scaled = np.stack((x, y, z))
	#scaler.fit(dataset_scaled)
	dataset_rescaled = scaler.inverse_transform(dataset_scaled.T)
	x_rescaled = np.zeros((N,N))
	x_rescaled = np.reshape(dataset_rescaled[:,0], (N,N))
	y_rescaled = np.zeros((N,N))
	y_rescaled = np.reshape(dataset_rescaled[:,1], (N,N))
	z_rescaled = np.zeros((N,N))
	z_rescaled = np.reshape(dataset_rescaled[:,2], (N,N))
	return x_rescaled, y_rescaled, z_rescaled
