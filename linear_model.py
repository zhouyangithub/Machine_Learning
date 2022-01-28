import numpy as np
from sklearn import datasets 
def linear_loss(X,y,w,b):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = X@w+b
    loss = (y_hat-y)@(y_hat-y)/num_train
    dw = np.dot(X.T,(y_hat-y))/num_train
    db = np.sum((y_hat-y))/num_train
    return y_hat,loss,dw,db
def initialize_params(dims):
    w = np.zeros((dims,1))
    b = 0
    return w,b
def linear_train(X,y,learning_rate = 0.01,epochs = 10000):
    loss_his = []
    w,b = initialize_params(X.shape[1])
    for i in range(1,epochs):
        y_hat,loss,dw,db = linear_loss(X,y,w,b)
        w -= learning_rate * dw
        b -= learning_rate * db
        loss_his.append(loss)
        if i % 10000 == 0:
            print("eposh %d loss %f"%(i,loss))
        params = {
            "w":w,
            "b":b
        }
        grads = {
            "dw":dw,
            "db":db
        }
    return loss_his,params,grads

from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
diabetes = load_diabetes()
data,target = diabetes.data,diabetes.target
X,y = shuffle(data,target,random_state=13)
offset = int(X.shape[0]*0.8)
X_train,y_train = X[:offset],y[:offset]
X_test,y_test = X[offset:],y[offset:]
print("X_train's shape:",X_train.shape)
print("X_test's shape:",X_test.shape)
print("y_train's shape:",y_train.shape)
print("y_test's shape:",X_test.shape)
loss_hit,params,grads = linear_train(X_train,y_train,0.01,200000)
print(params)