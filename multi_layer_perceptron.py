from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

#feature scaling on the training and test set

sc = MinMaxScaler()
sc.fit(xx_train)
xx_train = sc.transform(xx_train)
xx_test = sc.transform(xx_test)

#initializing the model and setting up the parameters

'''model structure :(Three hidden layers having 75 neurons for each,
activation function for all hidden layers : rectifier(relu),
epoch value: 200, optimizer function : lbfgs, learning rate : 0.003 and batch size : 28)'''

nn = MLPRegressor(hidden_layer_sizes=(75, 75, 75, ), activation='relu', max_iter= 200, solver='lbfgs', n_iter_no_change=100, learning_rate_init=0.003, batch_size=28)

#fitting training dataset to the model

nn.fit(xx_train, yy_train)

#prediction on test data

predictions = nn.predict(xx_test)