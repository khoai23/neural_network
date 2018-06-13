import tensorflow as tf

def createRandomArray(size):
	if(not isinstance(size, tuple) and not isinstance(size, list)):
		# incorrect shape, creating
		size = (1, size)
	array = tf.random_normal(size)
	return array

def createTensorflowSession(inputSize, outputSize, prefix='', learning_rate=1.0, hiddenLayers=[256], existedSession=None):
	# Create a session with specified values, return it at the end
	# Do not work for single layer perceptron (hiddenLayers empty)
	training_inputs = tf.placeholder(shape=[None, inputSize], dtype=tf.float32, name=prefix+'input')
	training_outputs = tf.placeholder(shape=[None, outputSize], dtype=tf.float32, name=prefix+'output')
	
	# Initialize weights and bias for hiddenLayers
	for i in range(len(hiddenLayers)):
		layerSize = hiddenLayers[i]
		if(i==0):
			prevLayer = training_inputs
			prevLayerSize = inputSize
		
		weights = tf.Variable(initial_value=createRandomArray((prevLayerSize, layerSize)), dtype=tf.float32, name=prefix+'W{}'.format(i))
		bias = tf.Variable(initial_value=createRandomArray(layerSize), dtype=tf.float32, name=prefix+'B{}'.format(i))
		# Initialize sum
		af_input = tf.matmul(prevLayer, weights) + bias
		# Use activation function
		currentLayer = tf.nn.sigmoid(af_input)
			
		# Prepare for the next iteration
		prevLayer = currentLayer
		prevLayerSize = layerSize
		
		if(i==len(hiddenLayers)-1):
			# Final value to output
			weights = tf.Variable(initial_value=createRandomArray((layerSize, outputSize)), dtype=tf.float32, name = prefix+'W{}'.format(i+1))  
			bias = tf.Variable(initial_value=createRandomArray(outputSize), dtype=tf.float32, name=prefix+'B{}'.format(i+1))
			# Initialize sum
			af_input = tf.matmul(currentLayer, weights) + bias
			# Use activation function to create prediction value
			prediction = tf.nn.sigmoid(af_input, prefix+'prediction')
			# Prediction error
			# prediction_error = tf.reduce_sum(training_outputs - prediction)
			prediction_error = 0.5 * tf.reduce_mean(tf.square(tf.subtract(training_outputs, prediction)))
			tf.identity(prediction_error, prefix+'prediction_error')
			# prediction_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=training_outputs, logits=prediction))
			# Minimize the prediction error using gradient descent optimizer
			train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name=prefix+'GD').minimize(prediction_error)
			
	# Initialization completed, run session
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	# print("Created: ",train_op, prediction, training_inputs, training_outputs)
	return sess, train_op, prediction, training_inputs, training_outputs
	
def runTrainingForSession(session, varTuple, dataTuple, epoch=100):
	train_op, training_inputs, training_outputs = varTuple
	
	# Format data to run correctly
	if(isinstance(dataTuple, list)):
		tupleData = ([], [])
		for set in dataTuple:
			tupleData[0].append(set[0])
			tupleData[1].append(set[1])
		dataTuple = tupleData
	# Run epoch
	for step in range(epoch):
		session.run(fetches=[train_op], feed_dict={training_inputs: dataTuple[0], training_outputs: dataTuple[1]})

def runWorkingSession(session, varTuple, data):
	prediction, training_inputs = varTuple
	
	return session.run(fetches=prediction, feed_dict={training_inputs: data})

def runTestTrainingSession(session, varTuple, data, predErrName):
	_, prediction, training_inputs, training_outputs = varTuple
	prediction_error = session.graph.get_tensor_by_name(predErrName)
	
	return session.run(fetches=[prediction_error, prediction], feed_dict={training_inputs: data[0], training_outputs: data[1]})
	
def runTest():
	dataIn = [[255, 0, 0], [248, 80, 68], [0, 0, 255], [67, 15, 210]]
	dataOut = [[1], [1], [0], [0]]
	
	session, train_op, prediction, input, output = createTensorflowSession(3, 1, '', 1.0, [200, 100])
	session.run(tf.global_variables_initializer())
	
	#loadFromPath(session, 'C:\\Python\\data\\ff.ckpt')
	
	dataTuple = (dataIn, dataOut)
	for i in range(20):
		runTrainingForSession(session, (train_op, input, output), dataTuple, 100)
		
		testIn = [[255, 0, 0], [247, 81, 67], [0, 0, 255]]
		testOut = runWorkingSession(session, (prediction, input), testIn)
		
		print("Out({}): {}".format(i+1, testOut))
	# print(tf.get_default_graph().get_operation_by_name('ioSize'))
	# saveToPath(session, 'C:\\Python\\data\\ff_test.ckpt')
	session.close()

def saveToPath(session, path):	
	saver = tf.train.Saver()
	saver.save(session, path)
def loadFromPath(session, path):
	saver = tf.train.Saver()
	try:
		saver.restore(session, path)
	except tf.errors.NotFoundError:
		print("Skip the load process @ path {} due to file not existing.".format(path))
	return session