from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

n = 3

# Making the Design Matrix
X = np.ones((len(x),int((n+2)*(n+1)/2)))

# Our attempt
#for i in range(int((n+2)*(n+1)/2)):
#	for j in range(n):
#		for k in range(n-j):
#			X[:,i] = x**j*y**k
			#print(k)

# From Vala/Per
for i in range(1,n+1):
	q = int((i)*(i+1)/2)
	for k in range(i+1):
		X[:,q+k] = x**(i-k)*y**k

# Making the Frenke function
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)
#print(z)

beta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(z))

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
