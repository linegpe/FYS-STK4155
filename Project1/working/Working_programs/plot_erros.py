import numpy as np
import matplotlib.pyplot as plt


poly = np.linspace(1,5,5)

"""
alpha01 = np.loadtxt("MSE_alpha01.txt", skiprows=1)
train01  = alpha01[:,0]
test01 = alpha01[:,1]

alpha005 = np.loadtxt("MSE_alpha005.txt", skiprows=1)
train005  = alpha005[:,0]
test005 = alpha005[:,1]

alpha001 = np.loadtxt("MSE_alpha001.txt", skiprows=1)
train001  = alpha001[:,0]
test001 = alpha001[:,1]

fig, axs = plt.subplots(3,sharex=True)

axs[0].plot(poly,test01,label="Train",color="mediumvioletred")
axs[0].plot(poly,train01,label="Test",color="teal")
axs[0].grid()
axs[0].legend()
axs[0].set_title(r"a) $\alpha=0.1$",y=0,x=0.02,fontsize=12,loc="left")

axs[1].plot(poly,test005,color="teal")
axs[1].plot(poly,train005,color="mediumvioletred")
axs[1].grid()
axs[1].set(ylabel="MSE")
axs[1].set_title(r"b) $\alpha=0.05$",y=0,x=0.02,fontsize=12,loc="left")

axs[2].plot(poly,test001,color="teal")
axs[2].plot(poly,train001,color="mediumvioletred")
axs[2].grid()
axs[2].set(xlabel="Polynomial degree")
axs[2].set_title(r"c) $\alpha=0.01$",y=0,x=0.02,fontsize=12,loc="left")

plt.show()


alpha1 = np.loadtxt("R2_alpha01.txt", skiprows=1)
test1  = alpha1[:,0]
train1 = alpha1[:,1]

alpha01 = np.loadtxt("R2_alpha001.txt", skiprows=1)
test01  = alpha01[:,0]
train01 = alpha01[:,1]

alpha05 = np.loadtxt("R2_alpha005.txt", skiprows=1)
test05  = alpha05[:,0]
train05 = alpha05[:,1]

fig, axs = plt.subplots(3,sharex=True)

axs[0].plot(poly,test01,label="Test",color="mediumvioletred")
axs[0].plot(poly,train01,label="Train",color="teal")
axs[0].grid()
axs[0].set_ylim(0,1.05)
axs[0].legend(loc="lower right")
axs[0].set_title(r"a) $\alpha=0.01$",y=0.7,x=0.13,fontsize=12)

axs[1].plot(poly,test05,color="teal")
axs[1].plot(poly,train05,color="mediumvioletred")
axs[1].grid()
axs[1].set_ylim(0,1.05)
axs[1].set(ylabel=r"$R^2$")
axs[1].set_title(r"b) $\alpha=0.05$",y=0.7,x=0.13,fontsize=12)

axs[2].plot(poly,test1,color="teal")
axs[2].plot(poly,train1,color="mediumvioletred")
axs[2].grid()
axs[2].set_ylim(0,1.05)
axs[2].set(xlabel="Polynomial degree")
axs[2].set_title(r"c) $\alpha=0.1$",y=0.7,x=0.12,fontsize=12)

plt.show()
"""
"""
poly = np.linspace(0,12,12)

Errors_test001 = np.loadtxt("b_alpha001.txt", skiprows=1)
MSE001 = Errors_test001[:,0]
Var001 = Errors_test001[:,1]
Bias001 = Errors_test001[:,2]

Errors_test005 = np.loadtxt("b_alpha005.txt", skiprows=1)
MSE005 = Errors_test005[:,0]
Var005 = Errors_test005[:,1]
Bias005 = Errors_test005[:,2]

Errors_test01 = np.loadtxt("b_alpha01.txt", skiprows=1)
MSE01 = Errors_test01[:,0]
Var01 = Errors_test01[:,1]
Bias01 = Errors_test01[:,2]

fig, axs = plt.subplots(3,sharex=True)

axs[0].plot(poly,MSE001,label="MSE test",color="sienna")
axs[0].plot(poly,Var001,label="Var test",color="olivedrab")
axs[0].plot(poly,Bias001,label="Bias test",color="slateblue")
axs[0].grid()
axs[0].legend()
axs[0].set_title(r"$\alpha=0.01$",y=0.93)

axs[1].plot(poly,MSE005,label="MSE test",color="sienna")
axs[1].plot(poly,Var005,label="Var test",color="olivedrab")
axs[1].plot(poly,Bias005,label="Bias test",color="slateblue")
axs[1].grid()
axs[1].set(ylabel=r"Error")
axs[1].set_title(r"$\alpha=0.05$",y=0.93)

axs[2].plot(poly,MSE01,label="MSE test",color="sienna")
axs[2].plot(poly,Var01,label="Var test",color="olivedrab")
axs[2].plot(poly,Bias01,label="Bias test",color="slateblue")
axs[2].grid()
axs[2].set(xlabel="Polynomial degree")
axs[2].set_title(r"$\alpha=.01$",y=0.93)

plt.show()

"""

poly = np.linspace(0,13,13)

Errors_test001 = np.loadtxt("b_MSE_alpha001.txt", skiprows=1)
Test001 = Errors_test001[:,0]
Train001 = Errors_test001[:,1]

Errors_test005 = np.loadtxt("b_MSE_alpha005.txt", skiprows=1)
Test005 = Errors_test005[:,0]
Train005 = Errors_test005[:,1]

Errors_test01 = np.loadtxt("b_MSE_alpha01.txt", skiprows=1)
Test01 = Errors_test01[:,0]
Train01 = Errors_test01[:,1]

fig, axs = plt.subplots(3,sharex=True)

plt.title("Test")

axs[0].plot(poly,Test001,label="Test",color="deeppink")
axs[0].plot(poly,Train001,label="Train",color="limegreen")
axs[0].grid()
axs[0].legend()
axs[0].set_title(r"a) $\alpha=0.01$",y=0.7)

axs[1].plot(poly,Test005,label="Test",color="deeppink")
axs[1].plot(poly,Train005,label="Train",color="limegreen")
axs[1].grid()
axs[1].set(ylabel=r"Error")
axs[1].set_title(r"b) $\alpha=0.05$",y=0.7)

axs[2].plot(poly,Test01,label="Test",color="deeppink")
axs[2].plot(poly,Train01,label="Train",color="limegreen")
axs[2].grid()
axs[2].set(xlabel="Polynomial degree")
axs[2].set_title(r"c) $\alpha=.01$",y=0.7)

plt.show()
