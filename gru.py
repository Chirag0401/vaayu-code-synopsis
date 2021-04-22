import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = keras.models.Sequential()
model.add(keras.layers.GRU(30, input_shape = (1, X_train.shape[1:][1]),activation = 'relu', recurrent_activation = 'relu',return_sequences = True))
model.add(keras.layers.GRU(30, activation = 'relu', recurrent_activation = 'relu', return_sequences = True))
model.add(keras.layers.GRU(30, activation = 'relu', recurrent_activation = 'relu', return_sequences = True))
model.add(keras.layers.Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(0.001))

es_callback = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = False)
history = model.fit(X_train, y_train, validation_split = 0.1, epochs = 50, verbose = 0, batch_size = 28,
                    callbacks = [es_callback])

predictions = model.predict(X_test)
predictions = predictions.flatten()