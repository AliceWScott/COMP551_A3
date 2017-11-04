from sklearn.model_selection import KFold
import NN
import numpy as np 
import json
import sys


'''
	Loads the weights and biases from a json file.
'''
def load_model(filename):

	with open(filename) as f:
		model = json.load(f)

	cost_fn = getattr(sys.modules[__name__], data['cost_fn'])
	activation_fn = getattr(sys.modules[__name__], data['activation_fn'])

	nn = FeedForwardNN(network_size=data["network_size"], 
					   cost_fn=cost_fn, 
					   activation_fn=activation_fn,
					   reg_lambda=data["reg_lambda"],
					   epsilon=data["epsilon"])
	nn.biases = [np.array(b) for b in data["biases"]]
	nn.weights = [np.array(w) for w in data["weights"]]

	return nn

'''
	Save the NN model to a json file.
'''
def save_model(filename, model):

	with open(filename, 'w') as f:
		json.dump(model, f)

def loadFromFile():

	print 'loading sets from file...'

	X = np.loadtxt('./train_x.csv', delimiter=",")
	Y = np.loadtxt('./train_y.csv', delimiter=",")
	X_reshaped = X.reshape((50000, 4096)) / 255 #normalized
	x_train = X_reshaped[:40000,]
	x_test = X_reshaped[40000:50000,]
	y_train = Y[:40000]
	y_test = Y[40000:]

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	y_train = y_train.astype(np.int32)
	y_train = y_train.astype(np.int32)

	training_set = zip(x_train, y_train)
	test_set = zip(x_test, y_test)

	return training_set, test_set


if __name__ == "__main__":

	training_set, test_set = loadFromFile()

	training_set = training_set[300:2000]
	validation_set = training_set[:300]
	
	print 'starting training...'
	
	nn = NN.FeedForwardNN([4096, 200, 40], epsilon=3.0)
	nn.init_bias_and_weights()
	nn.mini_batch_gradient_descent(max_epochs=30, 
								   batch_size=10,
								   training_set=training_set,
								   validation_set=validation_set)
	model = nn.get_model()
	save_model(model.json, model)
	nn.predict(validation_set, getAccuracy=True)



	# kf = KFold(n_splits=3, random_state=None, shuffle=True)
	# for train_index, test_index in kf.split(X):
	# 	X_train, X_test = X[train_index], X[test_index]
	# 	Y_train, Y_test = Y[train_index], Y[test_index]

	