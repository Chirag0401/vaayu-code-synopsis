import numpy as np
import pandas as pd

filename = "..\data\merged_new.csv"
col_list = ['PM10', 'Dew Point Temperature', 'Wind Speed', 'Day No.', 'Pressure Station Level']
target_y = ['PM10']
test_size = 0.2

data = pd.read_csv(filename)
data_set = data.iloc[:2372, : ]

y = dataset[target_y][1:].values
X = dataset[col_list][:-1].values
train_length = round(X.shape[0] * (1 - test_size))
        
X_train = np.array(X[:train_length])
y_train = np.array(y[:train_length])
X_test = np.array(X[train_length:])
y_test = np.array(y[train_length:])