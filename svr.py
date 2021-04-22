from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#peform scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_fit = scaler_X.fit_transform(X_train)
y_fit = scaler_y.fit_transform(y_train.reshape(-1, 1))
        
#fitting on training data
regressor = SVR(kernel = 'rbf')
regressor.fit(X_fit, y_fit.ravel())

#prediction on test data
predictions = scaler_y.inverse_transform(regressor.predict(scaler_X.transform(X_test)))
predictions = predictions.flatten()