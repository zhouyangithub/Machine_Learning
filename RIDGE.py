import numpy as np
def l2_loss(X,y,w,b,alpha):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X,w)+b
    loss = np.sum((y_hat-y)**2)/num_train+alpha*(np.sum(np.square(w)))
    dw = np.dot(X.T,(y_hat-y))/num_train+2*alpha*w
    db = np.sum((y_hat-y))/num_train
    return y_hat,loss,dw,db

def initialize_params(dims):
    w = np.zeros(dims)
    b = 0
    return w,b

def ridge_train(X,y,learning_rate = 0.01,epochs = 1000):
    loss_his = []
    w,b = initialize_params(X.shape[1])
    for i in range(1,epochs):
        y_hat,loss,dw,db = l2_loss(X,y,w,b)
        
