import numpy as np 
def sign(x):
    if x >0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1
vec_sign = np.vectorize(sign)

def l1_loss(X,y,w,b,alpha = 1):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = X@w + b
    loss = np.sum((y_hat-y)**2)/num_train+np.sum(alpha*abs(w))
    dw = np.dot(X.T,(y_hat-y))/num_train+alpha*vec_sign(w) 
    db = np.sum((y_hat-y))/num_train
    return y_hat,loss,dw,db

def initialize_params(dims):
    w = np.zeros(dims)
    b = 0
    return w,b

def lasso_train(X,y,learning_rate = 0.1,epochs = 1000):
    loss_his = []
    w,b = initialize_params(X.shape[1])
    for i in range(1,epochs):
        y_hat,loss,dw,db = l1_loss(X,y,w,b) 
        w += -learning_rate*dw
        b += -learning_rate*db
        loss_his.append(loss)
        if i % 50 == 0:
            print("epoch %d loss %f" %(i,loss))
        params = {'w':w,'b':b}
        grads = {'dw':dw,'db':db}
    return loss_his,params,grads

from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
diabetes = load_diabetes()
data,target = diabetes.data,diabetes.target
X,y = shuffle(data,target,random_state=13)
offset = int(X.shape[0]*0.8)
X_train,y_train = X[:offset],y[:offset]
X_test,y_test = X[offset:],y[offset:]
loss_hit,params,grads = lasso_train(X_train,y_train,0.01,30000)
print(params)