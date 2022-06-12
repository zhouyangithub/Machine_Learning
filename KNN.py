import numpy as np 
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle

iris = datasets.load_iris()
X,y = shuffle(iris.data,iris.target,random_state=13)
