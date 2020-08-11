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
X,y = load_data()
theta = gradient_descent(X,y)
plt.scatter(X[:,1],y)
x= np.linspace(5,25,100)
y = theta[0]+x*theta[1]
plt.plot(x,y,label = "predict",c = 'r')
plt.legend()
plt.show()
print(theta)