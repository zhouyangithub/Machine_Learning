import numpy as np
def sigmod(x):
    z = 1/(1+np.exp(-x))
    return z

def initializ_params(dims):
    W = np.zeros(dims)
    b = 1
    return W,b

def logistic(X,y,W,b):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    a = sigmod(np.dot(X,W)+b)
    cost = -1/num_train * np.sum(y*np.log(a)+(1-y)*np.log(1-a))
    dW = np.dot(X.T,(a-y))/num_train
    db = np.sum(a-y)/num_train
    cost = np.squeeze(cost)
    return a,cost,dW,db

def logistic_train(X,y,learning_rate,epochs):
    W,b = initializ_params(X.shape[1])
    cost_list = []
    for i in range(epochs):
        a,cost,dW,db = logistic(X,y,W,b)
        W = W - learning_rate * dW
        b = b - learning_rate * db
        if i % 100 == 0:
            cost_list.append(cost)
            print(f"epoch {i} cost {cost}")
    params = {
        "W":W,
        "b":b
    }
    grads = {
        "dW":dW,
        "db":db
    }
    return cost_list,params,grads
def predict(X,params):
    y_pred = sigmod(np.dot(X,params["W"])+params["b"])
    