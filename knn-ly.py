import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pandas as pd
data = np.loadtxt("d:\\data_analysis\knn-test.csv",delimiter=",")
#训练集和测试集的划分
num = int(data.shape[0]*0.7)
X,Y = data[:,0],data[:,1]
x,y = shuffle(X,Y)
x_train,x_test,y_train,y_test = x[:num],x[num:],y[:num],y[num:]
x_train,x_test = x_train.reshape(-1,1),x_test.reshape(-1,1)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
judge = pd.DataFrame({"Prediction":knn.predict(x_test),"Targect":y_test})
judge
score = pd.crosstab(judge.Targect,columns = judge.Prediction)/np.array([[112,112,112],[73,73,73],[57,57,57]])
score.columns = ["1","2","3"]
score.index = ["1","2","3"]
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(score.loc[["3","2","1"]],cmap=cmap,annot=True)
plt.xlabel("Predictions")
plt.ylabel("Targects")
plt.savefig("d://data_analysis/score.png")
