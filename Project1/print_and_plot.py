# This program contains all plotting functions

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from data_processing import *

from statistical_functions import R2, MSE, ols_svd, BetaVar

def SurfacePlot_Param(x,y,z_original,z_predicted):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z_original, cmap=cm.CMRmap,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("MSE as a function of model complexity and $\lambda$", fontsize=12)
    ax.set_ylabel("$\lambda$", fontsize=12)
    ax.set_xlabel("Polynomial degree", fontsize=12)
    ax.set_zlabel("MSE", fontsize=12)
    ax.tick_params(axis='x', labelsize=8 ) 
    ax.tick_params(axis='y', labelsize=8 ) 
    ax.tick_params(axis='z', labelsize=8 ) 
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def SurfacePlot(x,y,z_original,z_predicted):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z_original, cmap=cm.ocean,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title("Fitted terrain data for optimal parameters", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_xlabel("x", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    ax.tick_params(axis='x', labelsize=8 ) 
    ax.tick_params(axis='y', labelsize=8 ) 
    ax.tick_params(axis='z', labelsize=8 ) 
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
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
    plt.figure(figsize=(6.5,4.5))
    plt.plot(ModelComplexity,training_data_error,color="teal",label=" Train")
    plt.plot(ModelComplexity,test_data_error,color = "mediumvioletred", label="Test")
    plt.xlabel("Polynomial degree", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title("MSE for the train sand test datasets vs model complesity, OLS", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
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

def Plot_MSE_Test_Train(param, MSError_test, MSError_train, regression_method, resampling, text, option):
    plt.figure(figsize=(6.5,4.5))
    plt.plot(param,MSError_test,label="Test",color="mediumvioletred")
    plt.plot(param,MSError_train,label="Train", color="teal")
    if(option =="complexity"):
        plt.xlabel("Polynomial degree",fontsize=12)
    elif(option == "lambda"):
        plt.xlabel("$\lambda$",fontsize=12)
        plt.xscale("log")
    plt.ylabel("MSE",fontsize=12)
    #plt.title("MSE for the train and test datasets vs "+text+", "+"\n"+regression_method+", "+resampling, fontsize=12)
    plt.title("MSE for the train and test datasets vs "+text+", "+"\n"+regression_method+", no cross-validation", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def Plot_Bias_Variance(param, MSError_test, Variance_test, Bias_test, regression_method, resampling, option):
    plt.figure(figsize=(6.5,4.5))
    plt.plot(param,Variance_test,label="Variance",color="orange")
    plt.plot(param,Bias_test,label="Bias",color="steelblue")
    plt.plot(param,MSError_test,label="MSE",color="mediumvioletred")
    if(option =="complexity"):
        plt.xlabel("Polynomial degree",fontsize=12)
    elif(option == "lambda"):
        plt.xlabel("$\lambda$",fontsize=12)
        plt.xscale("log")
    plt.ylabel("Estimated error",fontsize=12)
    plt.title("Bias-variance tradeoff for the test dataset, "+'\n'+regression_method+", "+resampling,fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def Plot_Betas(PolyDeg, betas, param, regression_method, resampling):
    for i in range(int((PolyDeg+2)*(PolyDeg+1)/2)):
        plt.plot(param,betas[:,i],"maroon", linewidth = 1.5)
    plt.xlabel("$\lambda$",fontsize=12)
    plt.ylabel(r"Number of $\beta$",fontsize=12)
    plt.title(r"Parameters $\beta$ for different $\lambda$, "+'\n'+regression_method+", "+resampling, fontsize=12)
    plt.xscale("log")
    plt.grid(True)
    plt.show()

def Plot_Franke_Test_Train(z_test,z_train, X_test, X_train, scaler, x, y, z_noise):
    abs_train = np.zeros(np.int(len(X_train[:,0])))
    ord_train = np.zeros(np.int(len(X_train[:,0])))
    for i in range(np.int(len(X_train[:,0]))):
        abs_train[i]=X_train[i,1]
        ord_train[i]=X_train[i,2]
    
    dataset_scaled = np.stack((abs_train, ord_train, z_train))
    dataset_rescaled = scaler.inverse_transform(dataset_scaled.T)
    abs_train_resc = dataset_rescaled[:,0]
    ord_train_resc = dataset_rescaled[:,1]
    z_train_resc = dataset_rescaled[:,2]
    
    
    abs_test = np.zeros(np.int(len(X_test[:,0])))
    ord_test = np.zeros(np.int(len(X_test[:,0])))
    for i in range(np.int(len(X_test[:,0]))):
        abs_test[i]=X_test[i,1]
        ord_test[i]=X_test[i,2]
    
    dataset_scaled = np.stack((abs_test, ord_test, z_test))
    dataset_rescaled = scaler.inverse_transform(dataset_scaled.T)
    abs_test_resc = dataset_rescaled[:,0]
    ord_test_resc = dataset_rescaled[:,1]
    z_test_resc = dataset_rescaled[:,2]
    
    fig = plt.figure(figsize=(15,5))
    
    axs = fig.add_subplot(1, 3, 1, projection='3d')
    surf = axs.plot_surface(x, y, z_noise, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"a) Franke's function with noise", fontsize=12)
    axs.set_xlabel("x", fontsize=12)
    axs.set_ylabel("y", fontsize=12)
    axs.set_zlabel("z", fontsize=12)
    axs.set_zlim(-0.1,1.2)
    #plt.colorbar(surf, shrink=0.5, aspect=20)
    
    axs = fig.add_subplot(1, 3, 2, projection='3d')
    axs.scatter(abs_train_resc, ord_train_resc, z_train_resc, color = "blue")
    axs.set_title(r"b) Train data", fontsize=12)
    axs.set_xlabel("x", fontsize=12)
    axs.set_ylabel("y", fontsize=12)
    axs.set_zlabel("z", fontsize=12)
    
    axs = fig.add_subplot(1, 3, 3, projection='3d')
    axs.scatter(abs_test_resc, ord_test_resc, z_test_resc, color = "red")
    axs.set_title(r"c) Test data", fontsize=12)
    axs.set_xlabel("x", fontsize=12)
    axs.set_ylabel("y", fontsize=12)
    axs.set_zlabel("z", fontsize=12)
    axs.set_zlim(-0.1,1.25)
    plt.show() 

def Plot_Beta_with_Err(beta, X, alpha, sigma):
    plt.figure(figsize=(6.5,4.5))
    x = np.arange(len(beta))
    beta_error = np.sqrt(BetaVar(X,alpha,sigma))
    plt.errorbar(x,beta,yerr=beta_error,marker="o",markersize=3,color="maroon",linestyle="dashed",linewidth=1,capsize=2)
    plt.xlabel(r"Number of $\beta$",fontsize=12)
    plt.ylabel(r"Value of $\beta$",fontsize=12)
    plt.title(r"Parameters $\beta$ for polynomial degree 5, OLS",fontsize=12)
    plt.grid(True)
    plt.show()

def Plot_R2_Terrain(param, regression_method):

    r2_cv = np.loadtxt("Files/f_R2_Ridge_cv.txt", skiprows=1)
    R2_test_cv  = r2_cv[:]

    r2 = np.loadtxt("Files/f_R2_Ridge.txt", skiprows=1)
    R2_test  = r2[:]

    plt.plot(param, R2_test_cv, label="8-fold cross-validation",color="indigo")
    plt.plot(param, R2_test, label="No resampling",color="orangered")
    plt.xlabel("Polynomial degree",fontsize=12)
    plt.ylabel("$R^2$",fontsize=12)
    plt.title("$R^2$ for the test dataset vs model complexity,"+"\n"+regression_method, fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()
    plt.show()

def Plot_R2_all_Terrain(param):

    r2_test_OLS = np.loadtxt("Files/OLS_r2_test.txt", skiprows=1)
    R2_test_OLS  = r2_test_OLS[:]

    r2_train_OLS = np.loadtxt("Files/OLS_r2_train.txt", skiprows=1)
    R2_train_OLS  = r2_train_OLS[:]

    r2_test_Ridge = np.loadtxt("Files/Ridge_r2_test.txt", skiprows=1)
    R2_test_Ridge  = r2_test_Ridge[:]

    r2_train_Ridge = np.loadtxt("Files/Ridge_r2_train.txt", skiprows=1)
    R2_train_Ridge  = r2_train_Ridge[:]

    r2_test_LASSO = np.loadtxt("Files/LASSO_r2_test.txt", skiprows=1)
    R2_test_LASSO  = r2_test_LASSO[:]

    r2_train_LASSO = np.loadtxt("Files/LASSO_r2_train.txt", skiprows=1)
    R2_train_LASSO  = r2_train_LASSO[:]

    plt.plot(param, R2_test_OLS, label="OLS test",color="lawngreen")
    plt.plot(param, R2_train_OLS, "--", label="OLS train",color="lawngreen")
    plt.plot(param, R2_test_Ridge, label="Ridge test",color="orangered")
    plt.plot(param, R2_train_Ridge, "--", label="Ridge train",color="orangered")
    plt.plot(param, R2_test_LASSO, label="LASSO test",color="deepskyblue")
    plt.plot(param, R2_train_LASSO, "--", label="LASSO train",color="deepskyblue")

    plt.xlabel("$\lambda$",fontsize=12)
    plt.ylabel("$R^2$",fontsize=12)
    plt.title("$R^2$ for the test and train datasets vs $\lambda$", fontsize=12)
    plt.grid(True)
    plt.xscale("log")
    plt.legend(fontsize=12)
    plt.show()
    plt.show()

#-------------------- a) Plotting MSE test and train for different alpha ------------------------
"""
MaxPolyDeg = 5
ModelComplexity =np.arange(0,MaxPolyDeg+1)

alpha01 = np.loadtxt("Files/a_MSE_alpha01.txt", skiprows=1)
train01  = alpha01[:,0]
test01 = alpha01[:,1]

alpha005 = np.loadtxt("Files/a_MSE_alpha005.txt", skiprows=1)
train005  = alpha005[:,0]
test005 = alpha005[:,1]

alpha001 = np.loadtxt("Files/a_MSE_alpha001.txt", skiprows=1)
train001  = alpha001[:,0]
test001 = alpha001[:,1]

fig, axs = plt.subplots(3,sharex=True,figsize=(5.5,6.5))

axs[0].plot(ModelComplexity,test01,label="Train",color="teal")
axs[0].plot(ModelComplexity,train01,label="Test",color="mediumvioletred")
axs[0].grid()
axs[0].set_ylim(-0.1,1.1)
axs[0].legend()
axs[0].set_title(r"MSE for different noise amplitudes, OLS", fontsize=12)
axs[0].text(0.05, 0.06, "a) $\u03B1$=0.1", fontsize=12)

axs[1].plot(ModelComplexity,test005,color="mediumvioletred")
axs[1].plot(ModelComplexity,train005,color="teal")
axs[1].grid()
axs[1].set_ylim(-0.1,1.1)
axs[1].set_ylabel("MSE", fontsize=12)
axs[1].text(0.05, 0.06, "b) $\u03B1$=0.05", fontsize=12)

axs[2].plot(ModelComplexity,test001,color="mediumvioletred")
axs[2].plot(ModelComplexity,train001,color="teal")
axs[2].grid()
axs[2].set_ylim(-0.1,1.1)
axs[2].set_xlabel("Polynomial degree", fontsize=12)
axs[2].text(0.05, 0.06, "c) $\u03B1$=0.01", fontsize=12)

plt.show()

"""

#-------------------- a) Plotting R2 test and train for different alpha ------------------------

"""
MaxPolyDeg = 5
ModelComplexity =np.arange(0,MaxPolyDeg+1)

alpha1 = np.loadtxt("Files/a_R2_alpha01.txt", skiprows=1)
test1  = alpha1[:,0]
train1 = alpha1[:,1]

alpha01 = np.loadtxt("Files/a_R2_alpha001.txt", skiprows=1)
test01  = alpha01[:,0]
train01 = alpha01[:,1]

alpha05 = np.loadtxt("Files/a_R2_alpha005.txt", skiprows=1)
test05  = alpha05[:,0]
train05 = alpha05[:,1]

fig, axs = plt.subplots(3,sharex=True,figsize=(5.5,6.5))

axs[0].plot(ModelComplexity,test01,label="Test",color="mediumvioletred")
axs[0].plot(ModelComplexity,train01,label="Train",color="teal")
axs[0].grid()
axs[0].set_ylim(-0.1,1.1)
axs[0].legend(loc="lower right")
axs[0].set_title(r"$R^2$ for different noise amplitudes, OLS", fontsize=12)
axs[0].text(0.05, 0.86, "a) $\u03B1$=0.1", fontsize=12)

axs[1].plot(ModelComplexity,test05,color="mediumvioletred")
axs[1].plot(ModelComplexity,train05,color="teal")
axs[1].grid()
axs[1].set_ylim(-0.1,1.1)
axs[1].set_ylabel(r"$R^2$", fontsize=12)
axs[1].text(0.05, 0.86, "b) $\u03B1$=0.05", fontsize=12)

axs[2].plot(ModelComplexity,test1,color="mediumvioletred")
axs[2].plot(ModelComplexity,train1,color="teal")
axs[2].grid()
axs[2].set_ylim(-0.1,1.1)
axs[2].set_xlabel("Polynomial degree", fontsize=12)
axs[2].text(0.05, 0.86, "c) $\u03B1$=0.01", fontsize=12)

plt.show()

"""

#-------------------- c) Cross-validation nr of k-folds -----------------------------------------

"""
MaxPolyDeg = 18
ModelComplexity =np.arange(0,MaxPolyDeg+1)

k5 = np.loadtxt("Files/c_k5.txt", skiprows=1)
testk5  = k5[:,0]
traink5 = k5[:,1]

k8 = np.loadtxt("Files/c_k8.txt", skiprows=1)
testk8  = k8[:,0]
traink8 = k8[:,1]

k10 = np.loadtxt("Files/c_k10.txt", skiprows=1)
testk10  = k10[:,0]
traink10 = k10[:,1]

fig, axs = plt.subplots(3,sharex=True,figsize=(5.5,6.5))

axs[0].plot(ModelComplexity,testk5,label="Test",color="mediumvioletred")
axs[0].plot(ModelComplexity,traink5,label="Train",color="teal")
axs[0].grid()
axs[0].set_ylim(-0.1,1.1)
axs[0].legend(loc="upper right")
axs[0].legend(loc="r", bbox_to_anchor=(0.95,1))
axs[0].set_title(r"MSE for test and train datasets"+"\n"+ "for different number of k-folds, OLS, cross-validation", fontsize=12)
axs[0].text(0.5, 0.8, "a) k=5", fontsize=12)

axs[1].plot(ModelComplexity,testk8,color="mediumvioletred")
axs[1].plot(ModelComplexity,traink8,color="teal")
axs[1].grid()
axs[1].set_ylim(-0.1,1.1)
axs[1].set_ylabel(r"MSE", fontsize=12)
axs[1].text(0.5, 0.8, "b) k=8", fontsize=12)

axs[2].plot(ModelComplexity,testk10,color="mediumvioletred")
axs[2].plot(ModelComplexity,traink10,color="teal")
axs[2].grid()
axs[2].set_ylim(-0.1,1.1)
axs[2].set_xlabel("Polynomial degree", fontsize=12)
axs[2].text(0.5, 0.8, "c) k=10", fontsize=12)

plt.show()
"""

#-------------------- b,d,e) Plotting MSE test and train for different alpha ------------------------
"""
MaxPolyDeg = 13
ModelComplexity =np.arange(0,MaxPolyDeg+1)

alpha1 = np.loadtxt("Files/e_alpha_0.1.txt", skiprows=1)
test1  = alpha1[:,0]
train1 = alpha1[:,1]

alpha01 = np.loadtxt("Files/e_alpha_0.01.txt", skiprows=1)
test01  = alpha01[:,0]
train01 = alpha01[:,1]

alpha05 = np.loadtxt("Files/e_alpha_0.05.txt", skiprows=1)
test05  = alpha05[:,0]
train05 = alpha05[:,1]

fig, axs = plt.subplots(3,sharex=True,figsize=(5.5,6.5))

axs[0].plot(ModelComplexity,test1,label="Test",color="mediumvioletred")
axs[0].plot(ModelComplexity,train1,label="Train",color="teal")
axs[0].grid()
axs[0].set_ylim(-0.1,1.3)
axs[0].legend(loc="upper right")
axs[0].set_title(r"MSE for test and train datasets"+"\n"+ "for different noise amplitudes, LASSO, bootstrap", fontsize=12)
axs[0].text(0.05, 1.10, "a) $\u03B1$=0.1", fontsize=12)

axs[1].plot(ModelComplexity,test05,color="mediumvioletred")
axs[1].plot(ModelComplexity,train05,color="teal")
axs[1].grid()
axs[1].set_ylim(-0.1,1.3)
axs[1].set_ylabel(r"MSE", fontsize=12)
axs[1].text(0.05, 1.10, "b) $\u03B1$=0.05", fontsize=12)

axs[2].plot(ModelComplexity,test01,color="mediumvioletred")
axs[2].plot(ModelComplexity,train01,color="teal")
axs[2].grid()
axs[2].set_ylim(-0.1,1.3)
axs[2].set_xlabel("Polynomial degree", fontsize=12)
axs[2].text(0.05, 1.10, "c) $\u03B1$=0.01", fontsize=12)

plt.show()
"""

#-------------------- b, d, e) Plotting MSE test and train for different N ------------------------
"""
MaxPolyDeg = 13
ModelComplexity =np.arange(0,MaxPolyDeg+1)

N10 = np.loadtxt("Files/b_MSE_N10.txt", skiprows=1)
test10  = N10[:,0]
train10 = N10[:,1]

N20 = np.loadtxt("Files/b_MSE_N20.txt", skiprows=1)
test20  = N20[:,0]
train20 = N20[:,1]

N30 = np.loadtxt("Files/b_MSE_N30.txt", skiprows=1)
test30  = N30[:,0]
train30 = N30[:,1]

fig, axs = plt.subplots(3,sharex=True,figsize=(5.5,6.5))

axs[0].plot(ModelComplexity,test10,label="Test",color="mediumvioletred")
axs[0].plot(ModelComplexity,train10,label="Train",color="teal")
axs[0].grid()
axs[0].set_ylim(-0.1,1.35)
axs[0].legend(loc="upper right")
axs[0].set_title(r"MSE for test and train datasets"+"\n"+ "for different number of data points, OLS, bootstrap", fontsize=12)
axs[0].text(0.05, 1.15, "a) N = 100", fontsize=12)

axs[1].plot(ModelComplexity,test20,color="mediumvioletred")
axs[1].plot(ModelComplexity,train20,color="teal")
axs[1].grid()
axs[1].set_ylim(-0.1,1.35)
axs[1].set_ylabel(r"MSE", fontsize=12)
axs[1].text(0.05, 1.15, "b) N = 400", fontsize=12)

axs[2].plot(ModelComplexity,test30,color="mediumvioletred")
axs[2].plot(ModelComplexity,train30,color="teal")
axs[2].grid()
axs[2].set_ylim(-0.1,1.35)
axs[2].set_xlabel("Polynomial degree", fontsize=12)
axs[2].text(0.05, 1.15, "c) N = 900", fontsize=12)

plt.show()
"""
#-------------------- b, d, e) Plotting MSE test and train for different test/train ------------------------
"""
MaxPolyDeg = 13
ModelComplexity =np.arange(0,MaxPolyDeg+1)

test_set10 = np.loadtxt("Files/e_test_train_10.txt", skiprows=1)
test10  = test_set10[:,0]
train10 = test_set10[:,1]

test_set20 = np.loadtxt("Files/e_test_train_20.txt", skiprows=1)
test20  = test_set20[:,0]
train20 = test_set20[:,1]

test_set50 = np.loadtxt("Files/e_test_train_50.txt", skiprows=1)
test50  = test_set50[:,0]
train50 = test_set50[:,1]

fig, axs = plt.subplots(3,sharex=True,figsize=(5.5,6.5))

axs[0].plot(ModelComplexity,test10,label="Test",color="mediumvioletred")
axs[0].plot(ModelComplexity,train10,label="Train",color="teal")
axs[0].grid()
axs[0].set_ylim(-0.1,1.35)
axs[0].legend(loc="upper right")
axs[0].set_title(r"MSE for test and train datasets"+"\n"+ "for different test/train fractions, LASSO, bootstrap", fontsize=12)
axs[0].text(0.05, 1.15, "a) test/train = 1/9", fontsize=12)

axs[1].plot(ModelComplexity,test20,color="mediumvioletred")
axs[1].plot(ModelComplexity,train20,color="teal")
axs[1].grid()
axs[1].set_ylim(-0.1,1.35)
axs[1].set_ylabel(r"MSE", fontsize=12)
axs[1].text(0.05, 1.15, "b) test/train = 1/4", fontsize=12)

axs[2].plot(ModelComplexity,test50,color="mediumvioletred")
axs[2].plot(ModelComplexity,train50,color="teal")
axs[2].grid()
axs[2].set_ylim(-0.1,1.35)
axs[2].set_xlabel("Polynomial degree", fontsize=12)
axs[2].text(0.05, 1.15, "c) test/train = 1/1", fontsize=12)

plt.show()
"""

#-------------------- d) Plotting MSE test and train for different lambda ------------------------

"""
MaxPolyDeg = 20
ModelComplexity =np.arange(0,MaxPolyDeg+1)

test_set1 = np.loadtxt("Files/d_MSE_l_0.001.txt", skiprows=1)
test1  = test_set1[:,0]
train1 = test_set1[:,1]

test_set2 = np.loadtxt("Files/d_MSE_l_0.01.txt", skiprows=1)
test2  = test_set2[:,0]
train2 = test_set2[:,1]

test_set3 = np.loadtxt("Files/d_MSE_l_0.1.txt", skiprows=1)
test3  = test_set3[:,0]
train3 = test_set3[:,1]

test_set4 = np.loadtxt("Files/d_MSE_l_1.txt", skiprows=1)
test4  = test_set4[:,0]
train4 = test_set4[:,1]

test_set5 = np.loadtxt("Files/d_MSE_l_10.txt", skiprows=1)
test5  = test_set5[:,0]
train5 = test_set5[:,1]

test_set6 = np.loadtxt("Files/d_MSE_l_100.txt", skiprows=1)
test6  = test_set6[:,0]
train6 = test_set6[:,1]

plt.figure(figsize=(6.5,4.5))
plt.plot(ModelComplexity,test1,label=r"$\lambda$=0.001",color="indigo")
plt.plot(ModelComplexity,test2,label=r"$\lambda$=0.01",color="mediumvioletred")
plt.plot(ModelComplexity,test3,label=r"$\lambda$=0.1",color="hotpink")
plt.plot(ModelComplexity,test4,label=r"$\lambda$=1",color="royalblue")
plt.plot(ModelComplexity,test5,label=r"$\lambda$=10",color="dodgerblue")
plt.plot(ModelComplexity,test6,label=r"$\lambda$=100",color="skyblue")
plt.xlabel("Polynomial degree",fontsize=12)
plt.ylabel("MSE",fontsize=12)
plt.ylim(-0.03,1.1)
plt.xlim(-0.5,20.5)
plt.title("MSE for the test dataset vs model complexity, "+'\n'+"Ridge"+", cross-validation", fontsize=12)
plt.grid(True)
plt.legend(loc="l", bbox_to_anchor=(0.1,0.55))
plt.show()

plt.show()
"""

#----------------------------Plotting 6 surfaces ----------------------------------
def PlotSurfacesTerrain(x_scaled, y_scaled, scaler, N, M):

    z_predict_file_6 = np.loadtxt("Files/f_LASSO_cv_10_0.001.txt", skiprows=1)
    z_predict_6  = z_predict_file_6[:]
    x_rescaled, y_rescaled, z_rescaled_6 = DataRescaling(x_scaled,y_scaled,z_predict_6,scaler,N,M)

    z_predict_file_8 = np.loadtxt("Files/f_LASSO_cv_10_0.01.txt", skiprows=1)
    z_predict_8  = z_predict_file_8[:]
    x_rescaled, y_rescaled, z_rescaled_8 = DataRescaling(x_scaled,y_scaled,z_predict_8,scaler,N,M)

    z_predict_file_10 = np.loadtxt("Files/f_LASSO_cv_10_0.1.txt", skiprows=1)
    z_predict_10  = z_predict_file_10[:]
    x_rescaled, y_rescaled, z_rescaled_10 = DataRescaling(x_scaled,y_scaled,z_predict_10,scaler,N,M)

    z_predict_file_12 = np.loadtxt("Files/f_LASSO_cv_10_1.txt", skiprows=1)
    z_predict_12  = z_predict_file_12[:]
    x_rescaled, y_rescaled, z_rescaled_12 = DataRescaling(x_scaled,y_scaled,z_predict_12,scaler,N,M)

    z_predict_file_14 = np.loadtxt("Files/f_LASSO_cv_10_10.txt", skiprows=1)
    z_predict_14  = z_predict_file_14[:]
    x_rescaled, y_rescaled, z_rescaled_14 = DataRescaling(x_scaled,y_scaled,z_predict_14,scaler,N,M)

    z_predict_file_16 = np.loadtxt("Files/f_LASSO_cv_10_100.txt", skiprows=1)
    z_predict_16  = z_predict_file_16[:]
    x_rescaled, y_rescaled, z_rescaled_16 = DataRescaling(x_scaled,y_scaled,z_predict_16,scaler,N,M)
    

    fig = plt.figure(figsize=(6,7))
    fig.suptitle('Fits of the terrain data for different $\lambda$,'+"\n"+"LASSO with cross-validation", fontsize=14)
    
    axs = fig.add_subplot(3, 2, 1, projection='3d')
    surf = axs.plot_surface(x_rescaled, y_rescaled, z_rescaled_6, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"a) $\lambda$ = 0.001", fontsize=12)
    axs.set_xlabel("x", fontsize=8)
    axs.set_ylabel("y", fontsize=8)
    axs.set_zlabel("z", fontsize=8)
    axs.tick_params(axis='x', labelsize=8 ) 
    axs.tick_params(axis='y', labelsize=8 ) 
    axs.tick_params(axis='z', labelsize=8 )
    axs.view_init(elev=20., azim=140) 
    #axs.set_zlim(-0.1,1.2)
    
    axs = fig.add_subplot(3, 2, 2, projection='3d')
    surf = axs.plot_surface(x_rescaled, y_rescaled, z_rescaled_8, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"b) $\lambda$ = 0.01", fontsize=12)
    axs.set_xlabel("x", fontsize=8)
    axs.set_ylabel("y", fontsize=8)
    axs.set_zlabel("z", fontsize=8)
    axs.tick_params(axis='x', labelsize=8 ) 
    axs.tick_params(axis='y', labelsize=8 ) 
    axs.tick_params(axis='z', labelsize=8 ) 
    axs.view_init(elev=20., azim=140) 
    #axs.set_zlim(-0.1,1.2)
    
    axs = fig.add_subplot(3, 2, 3, projection='3d')
    surf = axs.plot_surface(x_rescaled, y_rescaled, z_rescaled_10, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"c) $\lambda$ = 0.1", fontsize=12)
    axs.set_xlabel("x", fontsize=8)
    axs.set_ylabel("y", fontsize=8)
    axs.set_zlabel("z", fontsize=8)
    axs.tick_params(axis='x', labelsize=8 ) 
    axs.tick_params(axis='y', labelsize=8 ) 
    axs.tick_params(axis='z', labelsize=8 ) 
    axs.view_init(elev=20., azim=140) 
    #axs.set_zlim(-0.1,1.2)

    axs = fig.add_subplot(3, 2, 4, projection='3d')
    surf = axs.plot_surface(x_rescaled, y_rescaled, z_rescaled_12, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"d) $\lambda$ = 1", fontsize=12)
    axs.set_xlabel("x", fontsize=8)
    axs.set_ylabel("y", fontsize=8)
    axs.set_zlabel("z", fontsize=8)
    axs.tick_params(axis='x', labelsize=8 ) 
    axs.tick_params(axis='y', labelsize=8 ) 
    axs.tick_params(axis='z', labelsize=8 ) 
    axs.view_init(elev=20., azim=140) 
    #axs.set_zlim(-0.1,1.2)

    axs = fig.add_subplot(3, 2, 5, projection='3d')
    surf = axs.plot_surface(x_rescaled, y_rescaled, z_rescaled_14, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"e) $\lambda$ = 10", fontsize=12)
    axs.set_xlabel("x", fontsize=8)
    axs.set_ylabel("y", fontsize=8)
    axs.set_zlabel("z", fontsize=8)
    axs.tick_params(axis='x', labelsize=8 ) 
    axs.tick_params(axis='y', labelsize=8 ) 
    axs.tick_params(axis='z', labelsize=8 ) 
    axs.view_init(elev=20., azim=140) 
    #axs.set_zlim(-0.1,1.2)

    axs = fig.add_subplot(3, 2, 6, projection='3d')
    surf = axs.plot_surface(x_rescaled, y_rescaled, z_rescaled_16, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"f) $\lambda$ = 100", fontsize=12)
    axs.set_xlabel("x", fontsize=8)
    axs.set_ylabel("y", fontsize=8)
    axs.set_zlabel("z", fontsize=8)
    axs.tick_params(axis='x', labelsize=8 ) 
    axs.tick_params(axis='y', labelsize=8 ) 
    axs.tick_params(axis='z', labelsize=8 ) 
    axs.view_init(elev=20., azim=140) 
    #axs.set_zlim(-0.1,1.2)

    plt.show()

def PlotTerrain(terrain, y_mesh, x_mesh, z_new):
    
    fig = plt.figure(figsize=(10,6))
    fig.suptitle("Terrain over Norway, original file and compressed", fontsize=14)
    axs = fig.add_subplot(1, 2, 1)
    axs.imshow(terrain, cmap=cm.ocean)
    axs.set_title(r"a) Original terrain data", fontsize=12)
    axs.set_xlabel("x", fontsize=12)
    axs.set_ylabel("y", fontsize=12)
    
    axs = fig.add_subplot(1, 2, 2, projection='3d')
    surf = axs.plot_surface(y_mesh, x_mesh, z_new, cmap=cm.ocean,linewidth=0, antialiased=False)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    axs.set_title(r"b) Compressed terrain data", fontsize=12)
    axs.set_xlabel("x", fontsize=10)
    axs.set_ylabel("y", fontsize=10)
    axs.set_zlabel("z", fontsize=10)
    axs.tick_params(axis='x', labelsize=8 ) 
    axs.tick_params(axis='y', labelsize=8 ) 
    axs.tick_params(axis='z', labelsize=8 )
    
    plt.show()
    