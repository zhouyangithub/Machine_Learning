import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pandas as pd
data = np.loadtxt("d:\\data_analysis\knn-test.csv",delimiter=",")
df = pd.read_excel("动作汇总.xlsx")
category = pd.DataFrame({"name":df.columns[1:],"index":range(df.columns[1:].shape[0])})
df.iloc[:,1:].shape
category.index
df2 = df.iloc[:,1:]
df2.columns = list(category.index)
data = df2.melt().dropna()
data = data.loc[data.value!= '--']
num = int(data.shape[0]*0.7)
X,Y = np.array(data)[:,1],np.array(data)[:,0].astype("float")
x,y = shuffle(X,Y)
x_train,x_test,y_train,y_test = x[:num],x[num:],y[:num],y[num:]
x_train,x_test = x_train.reshape(-1,1),x_test.reshape(-1,1)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
judge = pd.DataFrame({"Prediction":knn.predict(x_test),"Targect":y_test})
score = pd.crosstab(judge.Targect,columns = judge.Prediction)/np.array(pd.crosstab(judge.Targect,judge.Targect)).diagonal().repeat(20).reshape(20,-1).T
score.columns = category.name
score.index = category.name
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
cmap = sns.diverging_palette(230, 20, as_cmap=True)
figure = plt.figure(figsize=(18,15))
sns.heatmap(score,cmap=cmap,annot=True)
plt.xlabel("Predictions")
plt.ylabel("Targects")
plt.savefig("score.png",dpi = 900)