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
    a = sigmod(X@W+b)
    cost = -1/num_train * np.sum(y*np.log(a)+(1-y)*np.log(1-a))
    dW = (X.T@(a-y).reshape(90))/num_train
    db = np.sum(a-y)/num_train
    cost = np.squeeze(cost)
    return a,cost,dW,db

def logistic_train(X,y,learning_rate,epochs):
    y = y.reshape(1,-1)
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
    y_pred = sigmod((X@params["W"])+params["b"])
    y_pred2 = [1 if i >0.5 else 0 for i in y_pred]
    return y_pred2

import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_classification
X,labels = make_classification(n_samples=100,n_features=2,n_redundant=0,n_informative=2,random_state=1,n_clusters_per_class=2)
rng = np.random.RandomState(2)
X += 2*rng.uniform(size = X.shape)
plt.scatter(X[labels == 0,0],X[labels == 0,1])
plt.scatter(X[labels == 1,0],X[labels == 1,1])
#plt.show()
offset = int(X.shape[0]*0.9)
X_train,y_train = X[:offset],labels[:offset]
X_test,y_test = X[offset:],labels[offset:]
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

cost_list,params,grads = logistic_train(X_train,y_train,0.01,1000)
print(params)
y_pred = predict(X_test,params)
print(y_pred)