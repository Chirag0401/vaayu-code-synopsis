from sklearn.ensemble import RandomForestRegressor

#fitting model on train data
regressor = RandomForestRegressor(n_estimators = 50)
regressor.fit(xx_train, yy_train)

#prediction on test data
predictions = regressor.predict(xx_test)