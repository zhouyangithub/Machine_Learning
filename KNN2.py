import numpy as np 
import matplotlib.pyplot as plt 
from collections import Counter
from sklearn import datasets
from sklearn.utils import shuffle

iris = datasets.load_iris()
X,y = shuffle(iris.data,iris.target,random_state = 13)
X = X.astype(np.float32)
