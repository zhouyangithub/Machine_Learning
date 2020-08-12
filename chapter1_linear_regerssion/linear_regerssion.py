#This is a scrpit for linear regerssion 
import numpy as np 
import matplotlib.pyplot as plt  
#load data
def load_data():
    data = np.loadtxt("D:\Machine_Learning\chapter1_linear_regerssion\data1.txt",delimiter=",")
    X = np.column_stack((np.ones(data.shape[0]),data[:,0]))
    y = data[:,1]
    return X,y
def gradient_descent(X,y,iteration = 400,alpha = 0.02):
    theta = np.ones(X.shape[1])
    m = X.shape[0]
    n = X.shape[1]
    for num in range(iteration):
        for j in range(n):
            theta[j]+=(alpha/m)*np.sum((y-X@theta)*X[:,j])
    return theta
def featureNormalize(X):
    mu,sigma = X[:,1].mean(),X[:,1].std(ddof = 1)
    X[:,1] = (X[:,1]-mu)/sigma
def drawFigure(X,y,theta):
    plt.scatter(X[:,1],y)
    a = np.column_stack((np.ones(100),np.linspace(X[:,1].min(),X[:,1].max(),100)))
    b = a@theta
    plt.plot(a[:,1],b,label = "predict",c = 'r')
    plt.legend()
    plt.show()
X,y = load_data()
featureNormalize(X)
theta = gradient_descent(X,y)
drawFigure(X,y,theta)
print(theta)