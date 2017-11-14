import numpy as np 

"""
	Heavily commented so I know what I was doing when I look at this code again
	 in the future. (Same reasoning for the ULTRA VERBOSE variable names)
	 Code inspired by the following tutorials:
	 http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
	 https://databoys.github.io/Feedforward/
	 http://neuralnetworksanddeeplearning.com/chap2.html

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
		epsilon: learning rate for gradient descent.
		reg_lambda: regularization factor for weights.
	'''
	def __init__(self, network_size, epsilon=0.5, reg_lambda=0.01):

		self.num_layers = len(network_size)
		self.network_size = network_size
		self.epsilon = epsilon
		self.reg_lambda = reg_lambda

	''' 
		Randomly initialize bias and weights matrices using standard normal distribution.
		We don't set biases for the input layer (index = 0) or weights for the output layer (index = -1)
		because they will not be used.
		Sets weights for the connections between neurons of layers x and y.
	'''
	def init_bias_and_weights(self):

		np.random.seed(0)
		self.biases = [np.random.randn(y, 1) for y in self.network_size[1:]] 
		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.network_size[:-1], self.network_size[1:])]

	
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
			neuron_act = self.tanh(z_l)
			activations.append(neuron_act)

		if set_activations:
			return weighted_inputs, activations
		else:
			return neuron_act

	'''
		Splits dataset into batches for gradient descent.
	'''
	def split_mini_batches(self, data, batch_size):
		
		batches = []
		for i in xrange(0, len(data), batch_size):
			batches.append(data[i:i+batch_size])
		return batches

	''' 
		Mini-batch gradient descent.
		For regular stochastic gradient descent, use batch_size=1
		For batch gradient descent, use batch_size=len(training_set)
	'''
	def fit(self, max_epochs, batch_size, training_set, validation_set=None):

		reg = 1 - self.epsilon * (self.reg_lambda / len(training_set))

		for e in xrange(0, max_epochs):

			np.random.shuffle(training_set)

			for batch in self.split_mini_batches(training_set, batch_size):	

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

			if validation_set != None: 
				print "Epoch", e, ":", self.getAccuracy(validation_set)


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
		cost = self.cross_entropy_error(
							act=activation_matrix[-1], 
							y=y, 
							z=weighted_inputs[-1], 
							outputDelta=True)

		# rate of change of cost w.r.t. weights
		# equal to the matrix multiplication of neuron output error of layer l  and neuron input for layer l-1 
		# requires transposing the activation matrix
		gradient_weights[-1] = np.dot(cost, activation_matrix[-2].transpose())
		gradient_biases[-1] = cost # rate of change of the cost w.r.t. bias, equal to cost delta


		# actual backpropagation
		# find error vectors starting from the last layer
		for l in range(self.num_layers - 2, 0, -1):
			z = weighted_inputs[l]
			deriv = self.derive_tanh(z)
			cost = np.dot(self.weights[l+1].transpose(), cost) * deriv
			gradient_weights[l] = np.dot(cost, np.array(activation_matrix[l-1]).transpose())
			gradient_biases[l] = cost

		return gradient_biases, gradient_weights

	
	'''
		Makes the predictions on the unlabelled test set.
		Returns the index of the max argument in the forward pass, which gets converted to proper value.
	'''
	def predict(self, test_data):

		y_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
		15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35,
		 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] 

		return [y_values[int(np.argmax(self.forward_pass(x)))] for x in test_data]


	'''
		Prints out validation/test set accuracy.
		The model's predictions are returned as indices with the argmax.
		(also known as one-hot encoding)
		To get the actual predictions, we need to convert to proper number.
	'''
	def getAccuracy(self, validation):

		y_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
		15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35,
		 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] 

		count = 0
		for x, y in validation:
			prediction = np.argmax(self.forward_pass(x))
			if y_values[prediction] == y: count += 1

		return float(count) / len(validation) * 100.0

	''' 
		C = - (1/n) * SUM[ y * ln(a) + (1 - y) * ln(1 - a)]
	'''
	def cross_entropy_error(self, act, y, z=None, outputDelta=False):
		
		if outputDelta:
			return (act - y) # output layer error vector
		else:
			sum_over = y * np.log(act) + (1 - y) * np.log(1 - act)
			return  -np.sum(np.nan_to_num(sum_over)) #prevents returning NAN

	''' 
		Activation Function
	'''
	def tanh(self, x):
		return np.tanh(x)

	''' 
		Returns the derivative of tanh.
	'''
	def derive_tanh(self, x):
		return 1.0 - self.tanh(x)**2



