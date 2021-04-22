from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#fitting to a second degree polynomial
poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X_train)     

lin_reg = LinearRegression()

#fit the prediction model on train data
lin_reg.fit(X_poly, y_train)

#predicting test data
X_transform = poly_reg.fit_transform(X_test)
predictions = lin_reg2.predict(X_transform)