import tensorflow as tf

def createRandomArray(size):
	if(not isinstance(size, tuple) and not isinstance(size, list)):
		# incorrect shape, creating
		size = (1, size)
	array = tf.random_normal(size)
	return array

def createTensorflowSession(inputSize, outputSize, prefix='', trainingRate=1.0, hiddenLayers=[256], existedSession=None):
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
			weights = tf.Variable(initial_value=createRandomArray((layerSize, outputSize)), dtype=tf.float32, name=prefix+'W{}'.format(i+1))  
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
			train_op = tf.train.GradientDescentOptimizer(learning_rate=trainingRate, name=prefix+'GD').minimize(prediction_error)
			
	# Initialization completed, run session
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	# print("Created: ",train_op, prediction, training_inputs, training_outputs)
	return sess, train_op, prediction, training_inputs, training_outputs

## Deal with CNN later
def createConvolutionalSession(inputSize, outputSize, prefix='', trainingRate=1.0, hiddenLayers=[256], existedSession=None):
	# Create a session with specified values, return it at the end
	# Do not work for single layer perceptron (hiddenLayers empty)
	training_inputs = tf.placeholder(shape=[None, inputSize], dtype=tf.float32, name=prefix+'input')
	training_outputs = tf.placeholder(shape=[None, outputSize], dtype=tf.float32, name=prefix+'output')
	
	raise Exception("Not initialized function: createConvolutionalSession")
	
	
	# Initialization completed, run session
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	# print("Created: ",train_op, prediction, training_inputs, training_outputs)
	return sess, train_op, prediction, training_inputs, training_outputs

def createCustomizedSession(settingDict):
	# Create a session with specified values in a settingDict
	# Replace createTensorflowSession later
	inputSize = settingDict['inputSize']
	outputSize = settingDict['outputSize']
	prefix = settingDict.get('prefix', '')
	trainingRate = settingDict.get('trainingRate', 1.0)
	hiddenLayers = settingDict.get('hiddenLayers', [256])
	existedSession = settingDict.get('existedSession', None)
	dropout = settingDict.get('dropout', None)
	if(isinstance(dropout, int) and dropout > 0.0):
		dropout_op = tf.placeholder_with_default(dropout, shape=(), name=prefix+'dropout')
		dropout = True
	elif(isinstance(dropout, tf.Tensor)):
		dropout_op = dropout
		dropout = True
	else:
		dropout = False
	
	activationFunc = settingDict.get('activation', 'sigmoid')
	lossFunc = settingDict.get('loss', 'reduce_mean')
	if(activationFunc == 'tanh'):	activationFunc = tf.nn.tanh
	elif(activationFunc == 'relu'):	activationFunc = tf.nn.relu
	else:	activationFunc = tf.nn.sigmoid
	finalActivationFunc = settingDict.get('final', activationFunc)
	
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
		if(i > 0 and dropout):
			af_input = tf.nn.dropout(af_input, 1.0 - dropout_op)
		# Use activation function
		currentLayer = activationFunc(af_input)
			
		# Prepare for the next iteration
		prevLayer = currentLayer
		prevLayerSize = layerSize
		
		if(i==len(hiddenLayers)-1):
			# Final value to output
			weights = tf.Variable(initial_value=createRandomArray((layerSize, outputSize)), dtype=tf.float32, name=prefix+'W{}'.format(i+1))  
			bias = tf.Variable(initial_value=createRandomArray(outputSize), dtype=tf.float32, name=prefix+'B{}'.format(i+1))
			# Initialize sum
			af_input = tf.matmul(currentLayer, weights) + bias
			if(dropout):
				af_input = tf.nn.dropout(af_input, 1.0 - dropout_op)
			# Use activation function to create prediction value
			prediction = finalActivationFunc(af_input, prefix+'prediction')
			# Prediction error
			# prediction_error = tf.reduce_sum(training_outputs - prediction)
			prediction_error = 0.5 * tf.reduce_mean(tf.square(tf.subtract(training_outputs, prediction)))
			tf.identity(prediction_error, prefix+'prediction_error')
			# prediction_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=training_outputs, logits=prediction))
			# Minimize the prediction error using gradient descent optimizer
			train_op = tf.train.GradientDescentOptimizer(learning_rate=trainingRate, name=prefix+'GD').minimize(prediction_error)
			
	# Initialization completed, run session
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	# print("Created: ",train_op, prediction, training_inputs, training_outputs)
	if(dropout):
		return sess, train_op, prediction, training_inputs, training_outputs, dropout_op
	else:
		return sess, train_op, prediction, training_inputs, training_outputs

def createEncoder(settingDict):
	prefix = settingDict.get('prefix', 'Encoder')
	cellType = settingDict.get('cellType', 'lstm')
	forgetBias = settingDict.get('forgetBias', 1.0)
	dropout = settingDict.get('dropout', 1.0)
	layerSize = settingDict.get('layerSize', 32)
	layerDepth = settingDict.get('layerDepth', 2)
	# trainingRate = settingDict.get('trainingRate', 1.0)
	# learningDecay = settingDict.get('learningDecay', None)
	bidirectional = settingDict.get('bidirectional', False)
	inputType = settingDict.get('inputType', 'default')
	inputSize = settingDict['inputSize']
	
	# CellType selection here
	cellType = tf.contrib.rnn.BasicLSTMCell
	# Dropout variable here
	if(isinstance(dropout, int)):
		dropout = tf.placeholder_with_default(dropout, shape=(), name='dropout')
	elif(not isinstance(dropout, tf.Tensor)):
		raise Exception('Dropout in wrong type, not tf.Tensor|tf.Operation|number')
	# Input variation here
	if(isinstance(inputType, (tf.Operation, tf.Tensor))):
		inputs = inputType
	else:
		inputs = tf.placeholder(shape=[None, inputSize], dtype=tf.float32, name=prefix+'_input')
	# Encoder construction here
	if(bidirectional):
		if(layerDepth % 2 != 0):
			raise Exception("Bidirectional network must have an even number of layers")
		layerDepth = layerDepth / 2
		
		forwardLayers = createRNNLayers(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_forward')
		backwardLayers = createRNNLayers(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_backward')
		
		outputs, state = tf.nn.bidirectional_dynamic_rnn(forwardLayers, backwardLayers, inputs, dtype=tf.float32)
		outputs = tf.concat(outputs, -1)
		if(layerDepth == 1):
			state = [state]
		else:
			allState = []
			for i in range(layerDepth):
				allState.extend([state[0][i], state[1][i]])
			state = tuple(allState)
	else:
		layers = createRNNLayers(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_rnn')
		
		outputs, state = tf.nn.dynamic_rnn(layers, inputs, dtype=tf.float32)
		
	return inputs, outputs, state, dropout
	
def createDecoder(settingDict):
	prefix = settingDict.get('prefix', 'decoder')
	isBareMode = settingDict.get('mode', True)
	# the batchSize/maximumDecoderLength must be a placeholder that is filled during inference
	batchSize = settingDict['batchSize']; maximumDecoderLength = settingDict['maximumDecoderLength']; outputEmbedding = settingDict['outputEmbedding']
	encoderState = settingDict['encoderState']; correctResult = settingDict['correctResult']; decoderOutputSize = settingDict['decoderOutputSize']
	correctResultLen = settingDict['correctResultLen']; startToken = settingDict['startTokenId']; endToken = settingDict['endTokenId']
	decoderTrainingInput = settingDict['decoderInput']
	cellType = settingDict.get('cellType', 'lstm')
	cellType = tf.contrib.rnn.BasicLSTMCell
	forgetBias = settingDict.get('forgetBias', 1.0)
	dropout = settingDict.get('dropout', 1.0)
	layerSize = settingDict.get('layerSize', decoderOutputSize)
	layerDepth = settingDict.get('layerDepth', 2)
	
	attentionMechanism = settingDict.get('attention', None)
	if(attentionMechanism is not None):
		# attention mechanism use encoder output as input?
		encoderOutput = settingDict['encoderOutput']
		encoderLengthList = settingDict['encoderLengthList']
		# attentionLayerSize = settingDict.get('attentionLayerSize', 1)
		if('luong' in attentionMechanism):
			attentionMechanism = tf.contrib.seq2seq.LuongAttention(layerSize, encoderOutput, memory_sequence_length=encoderLengthList, scale=('scaled' in attentionMechanism))
		elif('bahdanau' in attentionMechanism):
			attentionMechanism = tf.contrib.seq2seq.BahdanauAttention(layerSize, encoderOutput, memory_sequence_length=encoderLengthList, normalize=('norm' in attentionMechanism))
		else:
			attentionMechanism = None
	# Dropout must be a placeholder/operation by now, as encoder will convert it
	assert isinstance(dropout, (tf.Operation, tf.Tensor)) and isinstance(correctResult, (tf.Operation, tf.Tensor))
	# Decoder construction here
	decoderCells = createRNNLayers(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_rnn')
	if(attentionMechanism is not None):
		decoderCells = tf.contrib.seq2seq.AttentionWrapper(decoderCells, attentionMechanism, attention_layer_size=layerSize, name=prefix + '_attention_wrapper')
	# conversion layer to convert from the hiddenLayers size into vector size if they have a mismatch
	# Helper for training using the output taken from the encoder outside
	trainHelper = tf.contrib.seq2seq.TrainingHelper(decoderTrainingInput, tf.fill([batchSize], maximumDecoderLength))
	# Helper for feeding the output of the current timespan for the inference mode
	startToken = tf.fill([batchSize], startToken)
	if(isBareMode):
		# Helper feeding the output directly into next input
		inferHelper = CustomGreedyEmbeddingHelper(outputEmbedding, startToken, endToken, True)
	else:
		# Helper using the argmax output as input
		inferHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(outputEmbedding, startToken, endToken)
	# Projection layer
	projectionLayer = tf.layers.Dense(decoderOutputSize, name=prefix+'_projection')
	# create the initial state out of a clone
	initialState = decoderCells.zero_state(dtype=tf.float32, batch_size=batchSize).clone(cell_state=encoderState)
	# depending on the mode being training or infer, use either Helper as fit
	inferDecoder = tf.contrib.seq2seq.BasicDecoder(decoderCells, inferHelper, initialState, output_layer=projectionLayer)
	trainDecoder = tf.contrib.seq2seq.BasicDecoder(decoderCells, trainHelper, initialState, output_layer=projectionLayer)
	# Another bunch of stuff I don't understand. Apparently the outputs and state are being created automatically. Yay.
	inferOutput, inferDecoderState, _ = tf.contrib.seq2seq.dynamic_decode(inferDecoder, maximum_iterations=maximumDecoderLength)
	trainOutput, trainDecoderState, _ = tf.contrib.seq2seq.dynamic_decode(trainDecoder)
	trainLogits = trainOutput.rnn_output
	inferLogits = inferOutput.rnn_output
	'''if(decoderOutputSize != layerSize):
		trainLogits = tf.layers.dense(trainLogits, decoderOutputSize, name='shared_projection', reuse=None)
		inferLogits = tf.layers.dense(trainLogits, decoderOutputSize, name='shared_projection', reuse=True)'''
	# sample_id is the argmax of the output. It can be used during beam searches and training
	sample_id = trainOutput.sample_id
	# correctResult in bareMode are [batchSize, maximumDecoderLength, vectorSize] represent correct value expected.
	# correctResult in bareMode are [batchSize, maximumDecoderLength] represent correct ids expected.
	if(isBareMode):
		lossOp, crossent = createDecoderLossOperation(trainLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength)
		secondaryLossOp, secondaryCrossent = createDecoderLossOperation(inferLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength, True)
		return (trainLogits, inferLogits), (lossOp, secondaryLossOp), (trainDecoderState, inferDecoderState), (crossent, secondaryCrossent)
	else:
		lossOp, crossent = createSoftmaxDecoderLossOperation(trainLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength)
		secondaryLossOp, secondaryCrossent = createSoftmaxDecoderLossOperation(inferLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength, True)
		return (inferLogits, trainLogits), (lossOp, secondaryLossOp), (inferDecoderState, trainDecoderState), (tf.argmax(inferLogits, axis=2), tf.argmax(trainLogits, axis=2)), (crossent, secondaryCrossent)
	
def createSingleDecoder(isTrainingMode, settingDict):
	prefix = settingDict.get('prefix', 'decoder')
	attentionMechanism = settingDict.get('attention', None)
	# the batchSize/maximumDecoderLength must be a placeholder that is filled during inference
	batchSize = settingDict['batchSize']; maximumDecoderLength = settingDict['maximumDecoderLength']; outputEmbedding = settingDict['outputEmbedding']
	encoderState = settingDict['encoderState']; correctResult = settingDict['correctResult']; encoderOutputSize = settingDict['encoderOutputSize']
	correctResultLen = settingDict['correctResultLen']; startToken = settingDict['startTokenId']; endToken = settingDict['endTokenId']
	decoderTrainingInput = settingDict['decoderInput']
	cellType = settingDict.get('cellType', 'lstm')
	cellType = tf.contrib.rnn.BasicLSTMCell
	forgetBias = settingDict.get('forgetBias', 1.0)
	dropout = settingDict.get('dropout', 1.0)
	layerSize = settingDict.get('layerSize', encoderOutputSize)
	layerDepth = settingDict.get('layerDepth', 2)
	if(attentionMechanism is not None):
		# attention mechanism use encoder output as input?
		encoderOutput = settingDict['encoderOutput']
		attentionLayerSize = settingDict.get('attentionLayerSize', 1)
		if(attentionMechanism == 'luong'):
			attentionMechanism = tf.contrib.seq2seq.LuongAttention(layerSize, encoderOutput)
		elif(attentionMechanism == 'bahdanau'):
			attentionMechanism = tf.contrib.seq2seq.BahdanauAttention(layerSize, encoderOutput)
		else:
			attentionMechanism = None
	# Dropout must be a placeholder/operation by now, as encoder will convert it
	assert isinstance(dropout, (tf.Operation, tf.Tensor)) and isinstance(correctResult, (tf.Operation, tf.Tensor))
	# Decoder construction here
	decoderCells = createRNNLayers(cellType, layerSize, layerDepth, forgetBias, dropout)
	if(attentionMechanism is not None):
		decoderCells = tf.contrib.seq2seq.AttentionWrapper(decoderCells, attentionMechanism)
	# conversion layer to convert from the hiddenLayers size into vector size if they have a mismatch
	if(encoderOutputSize != layerSize):
		decoderCells = tf.contrib.rnn.OutputProjectionWrapper(decoderCells, encoderOutputSize)
	
	if(isTrainingMode):
		# Helper for training using the output taken from the encoder outside
		trainHelper = tf.contrib.seq2seq.TrainingHelper(decoderTrainingInput, tf.fill([batchSize], maximumDecoderLength))
		trainDecoder = tf.contrib.seq2seq.BasicDecoder(decoderCells, trainHelper, encoderState)
		trainOutput, trainDecoderState, _ = tf.contrib.seq2seq.dynamic_decode(trainDecoder)
		trainLogits = trainOutput.rnn_output
		lossOp, crossent = createDecoderLossOperation(trainLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength)
		return trainLogits, lossOp, trainDecoderState, crossent
	else:
		# Helper for feeding the output of the current timespan for the inference mode
		startToken = tf.fill([batchSize], startToken)
		inferHelper = CustomGreedyEmbeddingHelper(outputEmbedding, startToken, endToken, True)
		inferDecoder = tf.contrib.seq2seq.BasicDecoder(decoderCells, inferHelper, encoderState)
		inferOutput, inferDecoderState, _ = tf.contrib.seq2seq.dynamic_decode(inferDecoder, maximum_iterations=maximumDecoderLength)
		inferLogits = inferOutput.rnn_output
		#secondaryLossOp, secondaryCrossent = createDecoderLossOperation(inferLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength, True)
		return inferLogits, None, inferDecoderState, None
	
	#return (inferLogits, trainLogits), (lossOp, secondaryLossOp), (inferDecoderState, trainDecoderState), (crossent, secondaryCrossent)
	
	
def createOptimizer(settingDict):
	assert all(key in settingDict for key in ['mode', 'loss'])
	loss = settingDict['loss']
	mode = settingDict['mode']
	if(mode == 'sgd'):
		# optimizer = 
		trainingRate = settingDict['trainingRate']
		if(isinstance(trainingRate, float)):
			trainingRate = tf.constant(trainingRate, dtype=tf.float32, name='training_rate_default')
		# warmUp and decay scheme. Customize later
		if('globalStep' in settingDict):
			globalStep = settingDict['globalStep']
			assert isinstance(globalStep, tf.Variable)
		if('warmupTraining' in settingDict and 'globalStep' in locals()):
			warmupStep, warmupThreshold = settingDict['warmupTraining']
			warmupFactor = tf.exp(-2 / warmupStep)
			inverseDecay = warmupFactor**(tf.to_float(warmupStep - globalStep))
			trainingRate = tf.cond(globalStep < warmupThreshold ,
									lambda: inverseDecay * trainingRate,
									lambda: trainingRate)
		if('decayTraining' in settingDict and 'globalStep' in locals()):
			decayStep, decayThreshold, decayFactor = settingDict['decayTraining']
			# warmupFactor = tf.exp(-2 / decayStep)
			# decayFactor = warmupFactor**(tf.to_float(warmupStep - globalStep))
			trainingRate = tf.cond(globalStep >= decayThreshold,
									lambda: tf.train.exponential_decay(trainingRate,(globalStep - decayThreshold), decayStep, decayFactor, staircase=True),
									lambda: trainingRate)
		
		return tf.train.GradientDescentOptimizer(trainingRate)
	elif(mode == 'adam'):
		if('trainingRate' not in settingDict):
			return tf.train.AdamOptimizer()
		else:
			trainingRate = settingDict['trainingRate']
			return tf.train.AdamOptimizer(trainingRate)
	else:
		raise Exception("Optimizer not specified.")
	
def configureGradientOptions(optimizer, settingDict):
	assert all(key in settingDict for key in ['colocateGradient', 'clipGradient', 'globalStep', 'loss'])
	loss = settingDict['loss']
	colocateGradient = settingDict['colocateGradient']
	# The gradient of all params affected 
	# affectedParams = tf.trainable_variables()
	# gradients = tf.gradients(loss, affectedParams, colocate_gradients_with_ops=colocateGradient)
	gradients, affectedParams = zip(*optimizer.compute_gradients(loss))
	# The maximum value of gradient allowed
	gradientClipValue = settingDict['clipGradient']
	gradientClipValue, globalNorm = tf.clip_by_global_norm(gradients, gradientClipValue)
	zippedGradientList = list(zip(gradientClipValue, affectedParams))
	# print(zippedGradientList[0])
	globalStep = settingDict['globalStep']
	return optimizer.apply_gradients(zippedGradientList, global_step=globalStep)
	
def createDecoderLossOperation(logits, correctResult, sequenceLengthList, batchSize, maxUnrolling, extraWeightTowardTop=False):
	# the maximum unrolling and batchSize for the encoder during the entire batch. correctResult should be [batchSize, sentenceSize, vectorSize], hence [1] and [0]
	# maxUnrolling = correctResult.shape[1].value
	# softmax cross entrophy between the correctResult and the decoder's output is used for p distribution
	# crossent = tf.nn.softmax_cross_entropy_with_logits(labels=correctResult, logits=logits)
	subtract = tf.reduce_mean(tf.square(tf.subtract(correctResult, logits)), axis=2)
	# mask to only calculate loss on the length of the sequence, not the padding
	target_weights = tf.sequence_mask(sequenceLengthList, maxUnrolling, dtype=logits.dtype)
	# May not be the most efficient opperation, but I digress
	target_weights = tf.transpose(tf.transpose(target_weights) / tf.to_float(sequenceLengthList))
	# The top units will be extra weights, used for greedyEmbedding as their initial results are extremely important
	if(extraWeightTowardTop):
		unrollingMask = tf.range(4, 0, -4.0 / tf.to_float(maxUnrolling))
		target_weights = tf.multiply(target_weights, unrollingMask)
	# the loss function being the reduce mean of the entire batch
	loss = tf.reduce_sum(tf.multiply(subtract, target_weights, name="subtract")) / tf.to_float(batchSize)
	return loss, target_weights
	
def createSoftmaxDecoderLossOperation(logits, correctIds, sequenceLengthList, batchSize, maxUnrolling, extraWeightTowardTop=False):
	# softmax the logits and compare it with correctIds. the correctIds will converted to onehot upon use
	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correctIds, logits=logits)
	# subtract = tf.reduce_mean(tf.square(tf.subtract(correctResult, logits)), axis=2)
	# mask to only calculate loss on the length of the sequence, not the padding
	target_weights = tf.sequence_mask(sequenceLengthList, maxUnrolling, dtype=tf.float32)
	# May not be the most efficient opperation, but I digress
	target_weights = tf.transpose(tf.transpose(target_weights) / tf.to_float(sequenceLengthList))
	# The top units will be extra weights, used for greedyEmbeddingHelper as their initial results are extremely important
	if(extraWeightTowardTop):
		unrollingMask = tf.range(4, 0, -4.0 / tf.to_float(maxUnrolling))
		target_weights = tf.multiply(target_weights, unrollingMask)
	# the loss function being the reduce mean of the entire batch
	loss = tf.reduce_sum(tf.multiply(crossent, target_weights, name="crossent")) / tf.to_float(batchSize)
	return loss, target_weights
	
def createRNNLayers(cellType, layerSize, layerDepth, forgetBias, dropout=None, name='RNN'):
	layers = []
	for i in range(layerDepth):
		cell = cellType(layerSize, forget_bias=forgetBias, name=name)
		if(dropout is not None):
			cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=dropout)
		layers.append(cell)
		
	return tf.contrib.rnn.MultiRNNCell(layers)
	
class CustomGreedyEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
	# Current GreedyEmbeddingHelper is getting the argmax value as id to be fed into the next time step
	# override its functions
	def __init__(self, embedding, start_tokens, end_token, normalized_embedding=None, normalized_length=None):
		super(CustomGreedyEmbeddingHelper, self).__init__(embedding, start_tokens, end_token)
		if(callable(embedding) and normalized_embedding is not None):
			self._normalized_length = normalized_length
			self._normalized_embedding = normalized_embedding
			self._using_normalized = False
		elif(not callable(embedding) and normalized_embedding is True):
			self._normalized_length = tf.norm(embedding, axis=1)
			self._normalized_embedding = embedding / tf.reshape(self._normalized_length, [tf.shape(self._normalized_length)[0], 1])
			self._using_normalized = False
		elif(not callable(embedding)):
			self._using_normalized = True
			self._embedding = embedding
		else:
			raise TypeError("Embedding is callable despite being normalized. Please use the constant version")
	
	def sample(self, time, outputs, state, name=None):
		"""sample for GreedyEmbeddingHelper."""
		del time, state  # unused by sample_fn
		# Outputs are logits, use argmax to get the most probable id
		if not isinstance(outputs, tf.Tensor):
			raise TypeError("Expected outputs to be a single Tensor, got: %s" % type(outputs))
		if(self._using_normalized):
			difference = tf.matmul(self._embedding, tf.transpose(outputs))
		else:
			output_len = tf.norm(outputs)
			output_normalized = outputs / output_len
			len_similarity = 1 - tf.tanh(tf.abs(tf.log(self._normalized_length / output_len)))
			arc_similarity = tf.matmul(self._normalized_embedding, tf.transpose(output_normalized))
			difference = len_similarity * tf.transpose(arc_similarity)
		sample_ids = tf.argmax(difference, axis=-1, output_type=tf.int32)
		return sample_ids
		
	def next_inputs(self, time, outputs, state, sample_ids, name=None):
		finished = tf.equal(sample_ids, self._end_token)
		next_inputs = tf.cond(time > 0, lambda: outputs, lambda: self._embedding_fn(sample_ids))
		return finished, next_inputs, state

def createHashDict(hashType, keyValueTensorOrTypeTuple=None, defaultValue=-1, inputTensor=None):
	if(inputTensor is None):
		inputTensor = tf.placeholder(tf.int32)
		
	if(hashType == 'HashTable' or hashType is tf.HashTable):
		hashType = tf.HashTable
		if(not isinstance(keyValueTensorOrTypeTuple, tf.KeyValueTensorInitializer)):
			raise Exception("keyValueTensorOrTypeTuple incorrect(not keyvaltensor) for static HashTable.")
		keyValueTensor = keyValueTensorOrTypeTuple
		
		table = hashType(keyValueTensor, defaultValue)
		return inputTensor, table.lookup(inputTensor)
	elif(hashType == 'MutableHashTable'):
		if(not isinstance(keyValueTensorOrTypeTuple, (tuple, list))):
			raise Exception("keyValueTensorOrTypeTuple incorrect (not tuple) for MutableHashTable.")
		keyType, valueType = keyValueTensorOrTypeTuple
		
		table = tf.contrib.lookup.MutableHashTable(key_dtype=keyType, value_dtype=valueType, default_value=defaultValue)
		return inputTensor, table.lookup(inputTensor)

def createEmbeddingSessionCBOW(dictSize, embeddingSize, inputSize=1, batch_size=512, trainingRate=1.0, existedSession=None, useNceLosses=True, numSample=9):
	# if use nce, create a lookup matrix and use default nce_loss function from tensorflow
	training_inputs = tf.placeholder(tf.int32, shape=[batch_size, inputSize])
	training_outputs = tf.placeholder(tf.int32, shape=[batch_size, 1])
	embeddings = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize)), dtype=tf.float32, name="E")
	embed = tf.nn.embedding_lookup(embeddings, training_inputs)
	# tensor is being properly squashed from [batch_size, embeddingSize, inputSize] into [batch_size, embeddingSize*inputSize]
	#embed = tf.div(tf.reduce_sum(embed, 1), inputSize)
	embed = tf.reshape(embed, [batch_size, embeddingSize*inputSize])

	# Construct the variables for the NCE loss
	nce_weights = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize*inputSize)), dtype=tf.float32, name="EW")
	nce_biases = tf.Variable(tf.zeros([dictSize]))

	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each
	# time we evaluate the loss.
	loss = tf.reduce_mean(tf.clip_by_value(
		tf.nn.nce_loss(weights=nce_weights,
						biases=nce_biases,
						labels=training_outputs,
						inputs=embed,
						num_sampled=numSample,
						num_classes=dictSize),
						1e-10, 100))
	train_op = tf.train.GradientDescentOptimizer(trainingRate).minimize(loss)
	
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	
	return sess, [train_op, loss], training_inputs, training_outputs, (embeddings, normalized_embeddings)
	
def createEmbeddingSession(dictSize, embeddingSize, inputSize=1, batch_size=512, trainingRate=1.0, existedSession=None, useNceLosses=True, numSample=9):
	# if use nce, create a lookup matrix and use default nce_loss function from tensorflow
	training_inputs = tf.placeholder(tf.int32, shape=[batch_size, inputSize])
	training_outputs = tf.placeholder(tf.int32, shape=[batch_size, 1])
	embeddings = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize)), dtype=tf.float32, name="E")
	embed = tf.nn.embedding_lookup(embeddings, training_inputs)
	# stopgap - tensor is being sum together
	#embed = tf.div(tf.reduce_sum(embed, 1), inputSize)
	embed = tf.reduce_mean(embed, 1)

	# Construct the variables for the NCE loss
	nce_weights = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize)), dtype=tf.float32, name="EW")
	nce_biases = tf.Variable(tf.zeros([dictSize]))

	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each
	# time we evaluate the loss.
	loss = tf.reduce_mean(tf.clip_by_value(
		tf.nn.nce_loss(weights=nce_weights,
						biases=nce_biases,
						labels=training_outputs,
						inputs=embed,
						num_sampled=numSample,
						num_classes=dictSize),
						1e-10, 100))
	train_op = tf.train.GradientDescentOptimizer(trainingRate).minimize(loss)
	
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	
	return sess, [train_op, loss], training_inputs, training_outputs, (embeddings, normalized_embeddings)
	
def runTrainingForSession(session, varTuple, dataTuple, epoch=100):
	if(len(dataTuple) == 0):
		return
	train_op, training_inputs, training_outputs = varTuple
	
	# Format data to run correctly
	if(isinstance(dataTuple, list)):
		tupleData = ([], [])
		for set in dataTuple:
			tupleData[0].append(set[0])
			tupleData[1].append(set[1])
		dataTuple = tupleData
	# Run epoch
	try:
		for step in range(epoch):
			session.run(fetches=[train_op], feed_dict={training_inputs: dataTuple[0], training_outputs: dataTuple[1]})
	except ValueError:
		print("ValueError @runTrainingForSession, data {}".format(dataTuple))
		#raise Exception("Error during training")

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
