from sklearn import metrics
import numpy as np

def get_performance(model, x_test, y_test):
    mae = metrics.mean_absolute_error(y_test, model.predict(x_test))#mean absolute error calculation
    mse = metrics.mean_squared_error(y_test, model.predict(x_test))#mean squared error calculation
    rmse = np.sqrt(metrics.mean_squared_error(y_test, model.predict(x_test)))#root mean squared error calculation
    rsq = metrics.r2_score(y_test, model.predict(x_test))#score calculation
    return (mae, mse, rmse, rsq)

#accuracy calculation(number of predictions having less than 20% error)
def get_accuracy(model, xx_test, yy_test):
    actual_val = yy_test
    predicted_val = model.predict(xx_test)
    predicted_val = predicted_val.reshape(predicted_val.shape[0], 1)
    error_val = predicted_val - actual_val
    count = []
    for i in range(len(actual_val)):
      if((abs(error_val[i])) < (0.2 * actual_val[i])):
        count.append(abs(error_val[i]))
    return(len(count) / len(error_val) * 100)