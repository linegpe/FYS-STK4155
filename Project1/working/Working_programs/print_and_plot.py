import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from statistical_functions import R2, MSE, ols_svd, BetaVar

def SurfacePlot(x,y,z_original,z_predicted):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z_original, cmap=cm.ocean,linewidth=0, antialiased=False)
    surf = ax.plot_surface(x, y, z_predicted, cmap=cm.Pastel1,linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def PrintErrors(z_train,z_tilde,z_test,z_predict):
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

def PlotErrors(ModelComplexity, training_data_error, test_data_error, error_type):
    plt.plot(ModelComplexity,training_data_error,label=error_type+" train")
    plt.plot(ModelComplexity,test_data_error,label=error_type+" test")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Estimated error")
    plt.legend()
    plt.show()

def WriteToFile(filename, method, z_train, z_predict, z_tilde, z_test):
    f = open(filename, "a")
    if (method=="MSE"):
        train = MSE(z_train,z_tilde)
        test = MSE(z_test,z_predict)
    elif (method=="R2"):
        train = R2(z_train,z_tilde)
        test = R2(z_test,z_predict)
    f.write("{0} {1} \n".format(float(train),float(test)))
    f.close()
