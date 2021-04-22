from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

#fit the prediction model on train data
lin_reg.fit(X_train, y_train)

#predicting test data
predictions = lin_reg.predict(X_test)