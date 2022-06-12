from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score

boston = load_boston()
X_train,X_test = boston.data[:int(0.7*boston.data.shape[0])],boston.data[int(0.7*boston.data.shape[0]):]
Y_train,Y_test = boston.target[:int(0.7*boston.data.shape[0])],boston.target[int(0.7*boston.data.shape[0]):]

regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)
y_pred = regr.predict(X_test)

print(f"Mean squared error: {mean_squared_error(y_pred,Y_test)}")
print(f"R Square score :{r2_score(y_pred,Y_test)}")
