import numpy as np 
def sign(x):
    if x >0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1
vec_sign = np.vectorize(sign)

def l1_loss(X,y,w,b,alpha):
    num_train = X.shape[0]
    num_feature = X.shape[1]
    y_hat = np.dot(X,w) + b
    loss = np.sum((y_hat-y)**2)/num_train+np.sum(alpha*abs(w))
    
