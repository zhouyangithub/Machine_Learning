from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
X_train,X_test = iris.data[:int(0.7*iris.data.shape[0])],iris.data[int(0.7*iris.data.shape[0]):]
Y_train,Y_test = iris.target[:int(0.7*iris.data.shape[0])],iris.target[int(0.7*iris.data.shape[0]):]

model = LogisticRegression()
model.fit(X_train,Y_train)

print(model.predict(X_test))
print(Y_test)