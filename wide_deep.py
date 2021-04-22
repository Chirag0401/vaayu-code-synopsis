import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

a_scaler = MinMaxScaler()
b_scaler = MinMaxScaler()

X_train_a = a_scaler.fit_transform(X_train_a)
X_train_b = b_scaler.fit_transform(X_train_b)

X_test_a = a_scaler.transform(X_test_a)
X_test_b = b_scaler.transform(X_test_b)

input_layer_a = keras.layers.Input(shape = X_train_a.shape[1:])
input_layer_b = keras.layers.Input(shape = X_train_b.shape[1:])

hidden1 = keras.layers.Dense(30, activation = 'relu')(input_layer_a)
hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)
hidden3 = keras.layers.Dense(30, activation = 'relu')(hidden2)

concat = keras.layers.concatenate([input_layer_b, hidden3])
output_layer = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs = [input_layer_a, input_layer_b], outputs = [output_layer])
model.compile(optimizer = keras.optimizers.Adam(0.001), loss = 'mse')

es_callback = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
history = model.fit((X_train_a, X_train_b), y_train, epochs = 50,
                    validation_split = 0.1, verbose = 0, batch_size = 28, callbacks = [es_callback])
train_predictions = model.predict((X_train_a, X_train_b))
predictions = model.predict((X_test_a, X_test_b))
