import numpy as np 

"""
	Heavily commented so I know wtf I was doing when I look at this code again
	 in the future. (Same reasoning for the ULTRA VERBOSE variable names)
"""
class FeedForwardNN:

	'''
		network_size (list) : length of the list indicates the length of the network.
							  The value at the i_th position indicates the number of neurons
							  in the i_th layer. The layer at index 0 is input, layer at index 
							  -1 is output.

		nodes in input layer = determined by dimensionality of our data
		nodes in output layer = 40, since there are 40 possible outputs 
			to single digit addition or multiplication

		activation (string): The activation function -- either 'sigmoid' or 'tanh'.
		cost (string): The cost function -- either 'cross_entropy' or 'sum_squared'
		epsilon: learning rate for gradient descent.
		reg_lambda: regularization factor for weights.
	'''
	def __init__(self, network_size, activation_fn='tanh', cost_fn='cross_entropy', epsilon=0.5, reg_lambda=0.01):

		self.num_layers = len(network_size)
		self.network_size = network_size
		self.epsilon = epsilon
		self.reg_lambda = reg_lambda

		if cost_fn == 'cross_entropy':
			self.cost_fn = self.cross_entropy_error
		elif cost_fn == 'sum_squared':
			self.cost_fn = self.sum_squared_error
		else:
			print 'ERROR: Not a valid cost function.'
			exit()

		if activation_fn == 'sigmoid':
			self.activation_fn = self.sigmoid
			self.activ_deriv = self.derive_sig
		elif activation_fn == 'tanh':
			self.activation_fn = self.tanh
			self.activ_deriv = self.derive_tanh
		else:
			print 'ERROR: Not a valid activation function.'
			exit()

	''' 
		Randomly initialize bias and weights matrices using standard normal distribution.
		We don't set biases for the input layer (index = 0) or weights for the output layer (index = -1)
		because they will not be used.
		Sets weights for the connections between neurons of layers x and y.
	'''
	def init_bias_and_weights(self):

		np.random.seed(0)
		# self.biases = [np.random.randn(y, 1) for y in self.network_size[1:]] 
		self.biases = [np.zeros((y,1)) for y in self.network_size[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.network_size[:-1], self.network_size[1:])]

	"""
		z : the weighted input
		outputDelta (Boolean): False if we are calculating cost for a single training
								example (C_x = (1/2)*(y - a_l)^2).
							   True if we are using the SSE partial derivative to find output layer error.
	"""
	def sum_squared_error(self, neuron_act, y, z=None, outputDelta=False):

		# output layer error vector = 
        #cost partial derivative w.r.t output activations * activation fn partial derivative w.r.t. weighted inputs
		if outputDelta and z != []:
			return (neuron_act - y) * self.activ_deriv(z)
		else:
			return 0.5 * (y - neuron_act)**2 #return cost

	''' 
		C = - (1/n) * SUM[ y * ln(a) + (1 - y) * ln(1 - a)]
	'''
	def cross_entropy_error(self, neuron_act, y, z=None, outputDelta=False):
		
		if outputDelta and z != []:
			return neuron_act - y # output layer error vector
		else:
			sum_over = y * np.log(neuron_act) + (1 - y) * np.log(1 - neuron_act)
			return  -np.sum(np.nan_to_num(sum_over)) #prevents returning NAN

	''' 
		The sigmoid activation function.
	'''
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	'''
		Calculates the derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
	'''
	def derive_sig(self, x):
		sig = self.sigmoid(x)
		return sig * (1.0 - sig)

	''' 
		Another option for activation function.
	'''
	def tanh(self, x):
		return np.tanh(x)

	''' 
		Returns the derivative of tanh.
	'''
	def derive_tanh(self, x):
		return 1.0 - self.tanh(x)**2

	""" 
		passes the NN's input into its matrix to return new activations
		neuron_act = matrix of neuron activations

		The activation a_l for the l_th layer is related to activation matrix in the (l-1)_th layer:
			a_l = sigma (weight_l * a_(l-1) + bias_l)
			where sigma is the activation function .

		The inner part of equation weight_l * a_(l-1) + bias_l is the weighted input of neurons in layer l (z_l).

	"""
	def forward_pass(self, neuron_act, set_activations=False):

		activations = []
		weighted_inputs = []

		for bias_vect, weight_vect in zip(self.biases, self.weights):
			z_l = np.dot(weight_vect, neuron_act) + bias_vect
			weighted_inputs.append(z_l)
			neuron_act = self.activation_fn(z_l)
			activations.append(neuron_act)

		if set_activations:
			return weighted_inputs, activations
		else:
			print neuron_act
			return neuron_act

	'''
		Vanilla Stochastic Gradient Descent implementation.
		Use mini-batch instead!!!!
	'''
	def stochastic_gradient_descent(self, max_epochs, training_set, validation_set=None):

		reg = 1 - self.epsilon * (self.reg_lambda / len(training_set))

		for e in xrange(0, max_epochs):
			np.random.shuffle(training_set)
			for example in training_set:
				self.evaluate_gradient([example])

	'''
		Computes gradient using the whole dataset.
		Only run if you've got massive amounts of time to kill.
	'''
	def batch_gradient_descent(self, max_epochs, training_set, validation_set=None):

		for e in xrange(0, max_epochs):
			np.random.shuffle(training_set)
			self.evaluate_gradient(training_set)		

	'''
		Splits dataset into batches for gradient descent.
	'''
	def split_mini_batches(self, data, batch_size):
		
		batches = []
		for i in xrange(0, len(data), batch_size):
			batches.append(data[i:i+batch_size])
		return batches

	''' 
		Mini batch gradient descent.
		The best option out of the 3 gradient descent functions.
	'''
	def mini_batch_gradient_descent(self, max_epochs, batch_size, training_set, validation_set=None):

		reg = 1 - self.epsilon * (self.reg_lambda / len(training_set))

		for e in xrange(0, max_epochs):
			np.random.shuffle(training_set)
			for batch in self.split_mini_batches(training_set, batch_size):				
				self.evaluate_gradient(batch, reg)



	''' 
		Updates the gradient's biases and weights.
		reg: Regularization factor to use when updating weight gradients.
	 '''
	def evaluate_gradient(self, batch, reg):

		#initialize matrices of zeros with size of biases and weights matrices
		gradient_biases = [np.zeros(b_vect.shape) for b_vect in self.biases]
		gradient_weights = [np.zeros(w_vect.shape) for w_vect in self.weights]

		#add in costs
		for x , y in batch:
			gradient_cost_biases, gradient_cost_weights = self.backpropagation(x,y)
			gradient_biases = [g + c for g, c in zip(gradient_biases, gradient_cost_biases)]
			gradient_weights = [g + c for g, c in zip(gradient_weights, gradient_cost_weights)]

		# bias = bias - (learning rate) * (loss w.r.t. parameters)
		updated_biases = []
		for b, g in zip(self.biases, gradient_biases): 
			bias = b - (self.epsilon / len(batch)) * g
			updated_biases.append(bias)
		self.biases = updated_biases

		# weight = regulization * weight - (learning rate) * (loss w.r.t. parameters)
		updated_weights = []
		for w, g in zip(self.weights, gradient_weights):
			weight = reg * w - (self.epsilon / len(batch)) * g
			updated_weights.append(weight)
		self.weights = updated_weights


	"""
		weighted_inputs stores the weighted input vectors z_l layer by layer

		cost for neuron j in layer l: 
			cost_j = (dC / dz_lj) where (dC / dz_lj) is the derivative of the Cost function w.r.t z_lj
	"""
	def backpropagation(self, X, y):

		#initialize matrices of zeros with size of biases and weights matrices
		gradient_biases = [np.zeros(b_vect.shape) for b_vect in self.biases]
		gradient_weights = [np.zeros(w_vect.shape) for w_vect in self.weights]

		# set the input layer activations
		weighted_inputs, activations = self.forward_pass(X, set_activations=True) 
		activation_matrix = [X] + activations 


		# calculate output layer error vector
		cost = self.cost_fn(activation_matrix[-1], y, z=weighted_inputs[-1], outputDelta=True)

		# rate of change of cost w.r.t. weights
		# equal to the matrix multiplication of neuron output error of layer l  and neuron input for layer l-1 
		# requires transposing the activation matrix
		cost_wrt_weight = np.dot(cost, activation_matrix[-2].transpose())

		# rate of change of the cost w.r.t. bias, equal to cost delta
		cost_wrt_bias = cost

		# actual backpropagation
		# find error vectors starting from the last layer
		for layer in xrange(2, self.num_layers):
			wi = weighted_inputs[-layer]
			deriv = self.activ_deriv(wi)
			cost = np.dot(self.weights[-layer+1].transpose(), cost) * deriv

			# print np.array(activation_matrix[-layer-1])
			gradient_weights[-layer] = np.dot(cost, np.array(activation_matrix[-layer-1]).transpose())
			gradient_biases[-layer] = cost

		return gradient_biases, gradient_weights


	''' 
		Makes the predictions on the validation or test set.
		Returns the index of the max argument in the forward pass.
		If using an unlabelled test set, hasYVals and printAccuracy must be set to False.
	 '''
	def predict(self, test_set, printAccuracy=False, hasYVals=False):
		
		results = []
		actual = []
		if hasYVals:
			for x,y in test_set:
				results.append(np.argmax(self.forward_pass(x)))
				actual.append(y)
		else:
			for x in test_set:
				results.append(np.argmax(self.forward_pass(x)))

		predictions = self.getPredictionValues(results)

		if printAccuracy and hasYVals:
			self.printAccuracy(zip(predictions, actual))

		return predictions


	'''
		Prints out validation/test set accuracy.
	'''
	def printAccuracy(self, results):

		count = 0
		for prediction, actual in results:
			if prediction == actual: count += 1

		print 'ACCURACY:', float(count) / len(results) * 100.0


	'''
		The model's predictions are returned as indices with the argmax.
		To get the actual predictions, we use this function.
	'''
	def getPredictionValues(self, results):
		
		y_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
		15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35,
		 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] 

		return [y_values[y] for y in results]

	'''
		Returns model's parameters as an object.
		Useful for saving the trained model to a JSON file.
	'''
	def get_model(self):

		return {"weights": [w.tolist() for w in self.weights],
				"biases": [b.tolist() for b in self.biases],
				"network_size": self.set_size,
				"cost_fn": str(self.cost_fn.__name__),
				"activation_fn": str(self.activation_fn.__name__),
				"epsilon": self.epsilon,
				"reg_lambda": self.reg_lambda
				}



