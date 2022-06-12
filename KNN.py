import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle

def compute_distance(X,X_train):
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test,num_train))
    M = np.dot(X,X_train.T)
    te = np.square(X).sum(axis = 1)
    tr = np.square(X_train).sum(axis = 1)
    dists = np.sqrt(-2*M+tr+np.matrix(te).T)
    return dists
#TODO:完成自己的距离计算函数
                     
def predict_labels(y_train,dists,k = 1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        labels = y_train[np.argsort(dists[i,:])].flatten()
        closest_y = labels[0:k]
        c = Counter(closest_y)
        y_pred[i] = c.most_common(1)[0][0]
    return y_pred

iris = datasets.load_iris()
X,y = shuffle(iris.data,iris.target,random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0]*0.7)
X_train,y_train = X[:offset],y[:offset]
X_test,y_test = X[offset:],y[offset:]
y_train,y_test = y_train.reshap((-1,1)),y_test.reshap((-1,1))
print(f"X_train = {X_train}\tX_test = {X_test}\ny_train = {y_train},y_test = {y_test}")

