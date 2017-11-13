from sklearn.model_selection import KFold
import NN
import numpy as np 
import json
import sys
from itertools import product
import pickle
import csv


'''
	Loads data from files, returns a list of tupled (x, y) values.
'''
def loadData(x_file, y_file):

	print 'loading sets from file...'

	X = np.loadtxt(x_file, delimiter=",")
	X = [np.reshape(x, (4096, 1)) for x in X]

	Y = np.loadtxt(y_file, delimiter=",")
	Y = Y.reshape(-1, 1)
	Y = Y.flatten()

	return zip(X,Y)

'''
	Performs cross-validation for hyperparameter tuning.
'''
def trainCV(training_set):

	np.random.shuffle(training_set)
	
	print 'starting training...'
	
	#using guidelines for choosing hidden neuron sizes from this stackexchange thread:
	# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw/1097#1097
	net_sizes = [[4096, 200, 40], [4096, 500, 40], [4096, 1000, 40], [4096, 2000, 40]]
	epsilon = [0.025, 0.25, 2.5]
	reg_lambda = [0.025, 0.25, 2.5, 5.0]
	num_epochs = [50, 100, 200]
	
	hyperparams = list(product(*[net_sizes, epsilon, reg_lambda, num_epochs]))
	best_params = (0.0, None)


	# tuning hyperparameters using cross-validation. 
	for h in hyperparams:

		print 'PARAMETERS:', h

		fold_accuracies = []
		current_model = None

		# Because of time constraints, we are tuning on only a very small subset of the training set.
		kf = KFold(n_splits=3, random_state=None, shuffle=True)
		for train_index, test_index in kf.split(training_set[:5000]):

			training_set = np.array(training_set)
			train, test = training_set[train_index], training_set[test_index]

			nn = NN.FeedForwardNN(h[0], epsilon=h[1], reg_lambda=h[2])
			nn.init_bias_and_weights()
			nn.mini_batch_gradient_descent(max_epochs=h[3], 
										   batch_size=16,
										   training_set=train,
										   validation_set=test)
			current_model = nn
			accuracy = nn.getAccuracy(zip(nn.predict(test, getAccuracy=True, hasYVals=True), [y for (x,y) in test]))
			fold_accuracies.append(accuracy)

		# get mean accuracy of all folds
		mean_acc = sum(fold_accuracies) / float(len(fold_accuracies))
		print 'CURRENT FOLD MEAN ACCURACY: ', mean_acc

		# if current index's mean accuracy is better than current best, update best_params and best_model
		if mean_acc > best_params[0]:
			best_params = (mean_acc, h)
			with open('best.pkl', 'wb') as f:
				pickle.dump(current_model, f)

	print 'THE BEST PARAMETERS ARE:', best_params[1][0], best_params[1][1], best_params[1][2], best_params[1][3]

# Write predictions to file.
def writeToFile(predictions):

	# Write predictions to file.
	with open('predictions.csv', 'w') as predictFile:
		fieldnames = ['Id', 'Label']
		writer = csv.DictWriter(predictFile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(1, len(predictions)):
			writer.writerow({'Id': i, 'Label': predictions[i]})

'''
	Make predictions for the unseen test set.
	Input is the filename for the test data.
'''
def predict_test_set(filename):

	#load data
	X = np.loadtxt(filename, delimiter=",")
	X = [np.reshape(x, (4096, 1)) for x in X]

	#load best model
	f = open('best.pkl', 'r')
	nn = pickle.load(f)

	# Predict on test set
	predictions = nn.predict(X)
	writeToFile(predictions)


if __name__ == "__main__":
	training_set = loadData('./train_x.csv', './train_y.csv')
	trainCV(training_set)
	predict_test_set('./test_x.csv')
