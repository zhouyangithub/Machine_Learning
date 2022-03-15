import numpy as np 
class LDA:
    def __init__(self):
        self.w = None
    def calc_cov(self,X,Y = None):
        m = X.shape[0]
        X = (X - np.mean(X,axis =0))/np.std(X,axis = 0)
        Y = X if Y == None else (Y - np.mean(Y,axis = 0))/np.std(Y,axis = 0)
        return 1/m * np.matmul(X.T,Y)
    def project(self,X,y):
        self.fit(X,y)
        