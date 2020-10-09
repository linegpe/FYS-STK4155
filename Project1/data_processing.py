import numpy as np
from sklearn.preprocessing import StandardScaler
from imageio import imread

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

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

def AddNoise(z,seed,alpha,N):
	np.random.seed(seed)
	noise = alpha*np.random.randn(N, N)
	z_with_noise = z+noise
	return z_with_noise

def DataScaling(x,y,z):
	# x and y must be raveled
	dataset = np.stack((x, y, z)).T
	scaler = StandardScaler()
	scaler.fit(dataset)
	[x_scaled, y_scaled, z_scaled] = scaler.transform(dataset).T
	return x_scaled, y_scaled, z_scaled, scaler

def DataRescaling(x,y,z,scaler,N,M):
	dataset_scaled = np.stack((x, y, z))
	#scaler.fit(dataset_scaled)
	dataset_rescaled = scaler.inverse_transform(dataset_scaled.T)
	x_rescaled = np.zeros((N,M))
	x_rescaled = np.reshape(dataset_rescaled[:,0], (N,M))
	y_rescaled = np.zeros((N,M))
	y_rescaled = np.reshape(dataset_rescaled[:,1], (N,M))
	z_rescaled = np.zeros((N,M))
	z_rescaled = np.reshape(dataset_rescaled[:,2], (N,M))
	return x_rescaled, y_rescaled, z_rescaled

def MinElement(MSError_test, R2_test, lambdas):
	min_el = np.min(MSError_test)
	min_el_arg = np.argmin(MSError_test)
	print(" Number of the minimum element: ",min_el_arg)
	print(" Lambda: ",lambdas[min_el_arg])
	print(" Minimum MSE: ",min_el)
	print(" R2: ", R2_test[min_el_arg])

def SetTerrainCompression(n):
	terrain = imread("SRTM_data_Norway_2.tif")
	terrain = np.delete(terrain,(0),axis=0)
	terrain = np.delete(terrain,(0),axis=1)

	# Extracting values
	N = np.size(terrain)
	nrows = np.size(terrain[:,0])
	ncols = np.size(terrain[0,:])
	x_mesh, y_mesh = np.meshgrid(range(int(ncols/n)),range(int(nrows/n)))
	z_mesh = terrain
	z_new = np.zeros((int(nrows/n),int(ncols/n)))

	x_flat = np.ravel(x_mesh)
	y_flat = np.ravel(y_mesh)

	# Compression
	for row in range(int(nrows/n)):
		for col in range(int(ncols/n)):
			z_new[row,col] = terrain[row*n,col*n]

	z_flat = np.ravel(z_new)

	N=len(z_new[:,0])
	M=len(z_new[0,:])
	return N, M, x_mesh, y_mesh, z_new, terrain, x_flat, y_flat, z_flat

