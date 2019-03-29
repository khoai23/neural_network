import tensorflow as tf
import numpy as np
import os

def _default_printer(*args):
	print(*args)

def optimizeLossClean(train_logits, correct_ids, correct_output_length, optimizer="SGD", learning_rate=1.0, decay_fn=None, clip_gradients=5.0, name="optimizer"):
	"""Create the optimize loss operator
		Args:
			...
		Returns:
			tuple of loss(scalar tensor) and train_op (tensorflow operation). train_op should already had increment global_step
	"""
	with tf.variable_scope(name):
		# entropy is [batch_size, tgt_len]
		entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=correct_ids, name="entropy")
		max_tgt_length = tf.shape(entropy)[1]
		with tf.control_dependencies([tf.assert_equal(max_tgt_length, tf.reduce_max(correct_output_length))]):
			# mask away the padding tokens
			mask = tf.sequence_mask(correct_output_length, maxlen=max_tgt_length, dtype=entropy.dtype, name="loss_mask")
			loss = entropy * mask
		# reduce to sentence loss (train) and word loss (token)
		loss = tf.reduce_sum(loss)
		token_loss = loss / tf.cast(tf.reduce_sum(correct_output_length), loss.dtype)
		train_loss = loss / tf.cast(tf.shape(correct_output_length)[0], loss.dtype)
		global_step = tf.train.get_or_create_global_step()
		return train_loss, token_loss, tf.contrib.layers.optimize_loss(train_loss, global_step, learning_rate, optimizer, learning_rate_decay_fn=decay_fn, clip_gradients=clip_gradients, name="backprop_op")

def getOrCreateEmbedding(name, create=False, vocab_size=None, num_units=None, scope="embeddings"):
	"""Get or create an embedding function converting ids to embedding
		Args:
			name: the name of the embedding
			create: if true, create the embedding with certainty
			vocab_size+num_units: the vocab an size of the embedding
		Returns:
			A lookup callable converting ids to embedding
	"""
	with tf.variable_scope(scope, reuse=not create):
		embedding = tf.get_variable(name, shape=[vocab_size, num_units] if create else None, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
	return lambda x: tf.nn.embedding_lookup(embedding, x)

def createEncoderClean(embedded_inputs, cell_size=512, cell_type=tf.nn.rnn_cell.BasicLSTMCell, num_layers=2, created_dropout=None, bidirectional=True, printer=_default_printer, name='encoder', concat_mode=False):
	"""Create an encoder (bi/uni)
		Args: 
			embedded_inputs: the embeddings of sentences, already in [batch_size, src_len, embedding_size]
			name: the name to use the scope in
			..: self explanatory
		Returns:
			Tuple of outputs [batch_size, src_len, embedding_size] and last encoder state (tuple|LSTMStateTuple)
	"""
	printer("Constructing encoder with setting values: {}".format(locals()))
	with tf.variable_scope(name):
		print("Current namescope {:s}".format(tf.get_variable_scope().name))
		# Check for dropout, create a placeholder if not yet created
		dropout = created_dropout
		if(not isinstance(dropout, tf.Tensor)):
			raise ValueError('Dropout in wrong type, not tf.Tensor|tf.Operation|number: {:s}'.format(created_dropout))
		# Construct the necessary encoder
		if(bidirectional):
			# if concat, split by cell_size; if stack, split by layers
			if(concat_mode):
				raise NotImplementedError("Concat mode not yet supported by base tf")
				if(cell_size % 2 != 0):
					raise ValueError("(CONCAT) Bidirectional network must have an even number of hidden units: {}".format(cell_size))
				cell_size = cell_size // 2
			else:
				if(num_layers % 2 != 0):
					raise ValueError("(STACK) Bidirectional network must have an even number of layers: {}".format(num_layers))
				num_layers = num_layers // 2
			
			forward_cell = createRNNCell(cell_type, cell_size, num_layers, dropout=dropout, name='forward_cell')
			backward_cell = createRNNCell(cell_type, cell_size, num_layers, dropout=dropout, name='backward_cell')
			
			outputs, state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, embedded_inputs, dtype=tf.float32)
			# this is stack mode
			outputs = tf.concat(outputs, -1)
			if(num_layers != 1):
				# with num_layers = 1, state is at correct shape and doesn't need to change
				state = state[0] + state[1]
		else:
			cell = createRNNCell(cell_type, cell_size, num_layers, dropout=dropout, name='encoder_cell')
			
			outputs, state = tf.nn.dynamic_rnn(cell, embedded_inputs, dtype=tf.float32)
		return outputs, state

def createDecoderClean(inputs, encoder_state, vocab_size, inputs_length=None, embedding_fn=None, projection_layer=None, encoder_outputs=None, encoder_length=None, cell_size=512, cell_type=tf.nn.rnn_cell.BasicLSTMCell, num_layers=2, created_dropout=None, attention_mechanism=None, printer=_default_printer, name='decoder', variable_scope_reuse=False, beam_size=1, training=False, end_token=None, maximum_iterations=250):
	"""Create a decoder with or without attention and beam
		If using attention, encoder_outputs and encoder_length required as tensor [batch_size, src_len, embedding_size] and [batch_size]
		If want to train|eval, set training=True and inputs is the correct inputs of the whole sentence; inputs_length required
		If want to infer, set training=False and inputs is the starting [batch_size] of start_token; embedding_fn, beam_size, end_token required
	Args: 
#DEP	projection_layer: Required in all, convert cell output to probs
			vocab_size: Required, the size of the projection. Replace to be self-contained
			inputs: see above
			variable_scope_reuse: set if this function had ran before
			training: set if in training mode
			attention mechanism: callable that create an attention object in which AttentionWrapper can use to create an attention cell
			..: self explanatory
		Returns:
			Tuple of outputs [batch_size, src_len, embedding_size] and last encoder state (tuple|LSTMStateTuple)
	"""
	printer("Constructing decoder with setting values: {}".format(locals()))
	if(not training):
		assert embedding_fn and callable(embedding_fn), "If not in training mode, must have the embedding function to do sequential decode"
		assert end_token is not None, "Must have a valid end_token value"
	else:
		assert inputs_length is not None, "Must have inputs_length in training"

	with tf.variable_scope(name, reuse=variable_scope_reuse):
		printer("Current namescope {:s}".format(tf.get_variable_scope().name))
	# with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
		projection_layer = tf.layers.Dense(vocab_size, use_bias=False, name="decoder_projection_layer")
		# construct cell
		cell = createRNNCell(cell_type, cell_size, num_layers, dropout=created_dropout, name='rnn_cell')
		if(attention_mechanism):
			assert callable(attention_mechanism), "Attention mechanism must be callable!"
			assert encoder_outputs is not None and encoder_length is not None, "Attention must have outputs and length!"
			# create the attention mechanism using the outputs and lengths
			if(not training and beam_size > 1):
				# tile the encoder states and outputs
				encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_size)
				encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_size)
				encoder_length = tf.contrib.seq2seq.tile_batch(encoder_length, multiplier=beam_size)
			attention = attention_mechanism(cell_size, encoder_outputs, encoder_length)
			# wrap the cell with the created attention
			cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention, initial_cell_state=encoder_state, attention_layer_size=cell_size, name="attention_wrapper", alignment_history=False)
			# the AttentionWrapper cell will create additional starter state in zero_state
			batch_size = tf.shape(encoder_outputs)[0]
			initial_state = cell.zero_state(batch_size, dtype=encoder_outputs.dtype)
		else:
			initial_state = encoder_state
		# initiate the decoding
		if(training):
			# Training decoder
			helper = tf.contrib.seq2seq.TrainingHelper(inputs, inputs_length)
			decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer=projection_layer)
			# run the dynamic decode
			outputs, final_state, final_length = tf.contrib.seq2seq.dynamic_decode(decoder)
			logits = outputs.rnn_output
			return { 
				"logits": logits, 
				"state": final_state, 
				"length": final_length 
			}
		elif(beam_size == 1):
			start_tokens = inputs
			# GreedyEmbeddingDecoder
			helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_fn, start_tokens, end_token)
			decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer=projection_layer)
			# run the dynamic decode
			outputs, final_state, final_length = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
			predictions = outputs.sample_id
			alignment_history = final_state.alignment_history if attention_mechanism else None
			return {
				"predictions": tf.expand_dims(predictions, 1),
				"state": final_state,
				"length": tf.expand_dims(final_length, -1),
				"log_probs": final_state,
				"alignment_history": alignment_history
			}
		else:
			# tile the inputs and run the BeamSearchDecoder
			start_tokens = inputs
#			start_tokens = tf.Print(start_tokens, [tf.shape(start_tokens), tf.shape(encoder_state), tf.shape(encoder_outputs), tf.shape(encoder_length)])
			decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, embedding_fn, start_tokens, end_token, initial_state, beam_size, output_layer=projection_layer)
			outputs, final_state, final_length = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
			# get the necessary values
			predictions = outputs.predicted_ids
#			alignment_history = final_state.alignment_history if attention_mechanism else Nonea
			alignment_history = None
			if(alignment_history):
				batch_size = tf.shape(inputs)[0]
				src_len = tf.shape(encoder_outputs)[1]
				alignment_history = tf.reshape(alignment_history, [batch_size, beam_size, -1, src_len])
			else:
				printer("Alignment history not available currently")
			return {
				"predictions": tf.transpose(predictions, perm=[0, 2, 1]),
				"state": final_state,
				"length": final_length,
				"log_probs": final_state.log_probs,
				"alignment_history": alignment_history
			}

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

def createEncoder(settingDict, printer=_default_printer):
	prefix = settingDict.get('prefix', 'encoder')
	cellType = settingDict.get('cellType', 'lstm')
	forgetBias = settingDict.get('forgetBias', 0.5)
	dropout = settingDict.get('dropout', 1.0)
	layerSize = settingDict.get('layerSize', 32)
	layerDepth = settingDict.get('layerDepth', 2)
	# trainingRate = settingDict.get('trainingRate', 1.0)
	# learningDecay = settingDict.get('learningDecay', None)
	bidirectional = settingDict.get('bidirectional', True)
	inputType = settingDict.get('inputType', 'default')
	inputSize = settingDict['inputSize']
	printer("Constructing encoder with setting values: {}".format(locals()))
	
	# CellType selection here
	cellType = tf.contrib.rnn.BasicLSTMCell
	# Dropout variable here
	if(isinstance(dropout, (int, float))):
		dropout = tf.placeholder_with_default(tf.to_float(dropout), shape=(), name='dropout')
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
		layerDepth = layerDepth // 2
		
		forwardLayers = createRNNCell(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_forward')
		backwardLayers = createRNNCell(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_backward')
		
		outputs, state = tf.nn.bidirectional_dynamic_rnn(forwardLayers, backwardLayers, inputs, dtype=tf.float32)
		outputs = tf.concat(outputs, -1)
		state = tuple([cellState[0] for cellState in state])
		#if(layerDepth > 1):
		#	state = [state]
		#else:
		#	allState = []
		#	for i in range(layerDepth):
		#		allState.extend((state[0][i], state[1][i]))
		#	state = tuple(allState)
	else:
		layers = createRNNCell(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_rnn')
		
		outputs, state = tf.nn.dynamic_rnn(layers, inputs, dtype=tf.float32)
		
	return inputs, outputs, state, dropout
	
def createDecoder(settingDict, printer=_default_printer):
	prefix = settingDict.get('prefix', 'decoder')
	# isBareMode = settingDict.get('mode', True)
	# the batchSize/maximumDecoderLength must be a placeholder that is filled during inference
	maximumDecoderLength = settingDict['maximumDecoderLength']; outputEmbedding = settingDict['outputEmbedding']
	encoderState = settingDict['encoderState']; correctResult = settingDict['correctResult']; decoderOutputSize = settingDict['decoderOutputSize']
	correctResultLen = settingDict['correctResultLen']; startToken = settingDict['startTokenId']; endToken = settingDict['endTokenId']
	batchSize = settingDict['batchSize']
	cellType = settingDict.get('cellType', 'lstm')
	cellType = tf.contrib.rnn.BasicLSTMCell
	forgetBias = settingDict.get('forgetBias', 1.0)
	dropout = settingDict.get('dropout', 1.0)
	layerSize = settingDict.get('layerSize', decoderOutputSize)
	layerDepth = settingDict.get('layerDepth', 2)
	beamSize = settingDict.get('beamSize', 1)
	printer("Constructing decoder with setting values: {}".format(locals()))
	
	assert correctResult is not None
	# the training input append startToken id to the front of all training sentences 
	correctShape = tf.shape(correctResult)
	decoderTrainingInput = tf.concat([tf.fill([correctShape[0], 1], startToken), correctResult], axis=1)
	# remove the last index in d2 (tf.Tensor.getittem)
	# decoderTrainingInput = tf.slice(decoderTrainingInput, [0, 0], [correctShape[0], correctShape[1] - 2])
	decoderTrainingInput = decoderTrainingInput[:, :-1]
	# look up the input through the outputEmbedding
	with tf.control_dependencies([tf.assert_equal(tf.shape(decoderTrainingInput), tf.shape(correctResult))]):
		decoderTrainingInput = tf.nn.embedding_lookup(outputEmbedding, decoderTrainingInput, name='input_decoder_vectors')
	
	attentionMechanism = settingDict.get('attention', None)
	if(attentionMechanism is not None):
		# attention mechanism use encoder output as input?
		encoderOutput = settingDict['encoderOutput']
		encoderLengthList = settingDict['encoderLengthList']
		if(beamSize > 1):
			tiledMemory = tf.contrib.seq2seq.tile_batch(encoderOutput, multiplier=beamSize)
			tiledLength = tf.contrib.seq2seq.tile_batch(encoderLengthList, multiplier=beamSize)
		# attentionLayerSize = settingDict.get('attentionLayerSize', 1)
		attentionMechanismName = attentionMechanism
		if('luong' in attentionMechanismName):
			attentionMechanism = tf.contrib.seq2seq.LuongAttention(layerSize, memory=encoderOutput, memory_sequence_length=encoderLengthList, scale=('scaled' in attentionMechanismName))
			if(beamSize > 1):
				beamAttentionMechanism = tf.contrib.seq2seq.LuongAttention(layerSize, memory=tiledMemory, memory_sequence_length=tiledLength, scale=('scaled' in attentionMechanismName))
		elif('bahdanau' in attentionMechanismName):
			attentionMechanism = tf.contrib.seq2seq.BahdanauAttention(layerSize, memory=encoderOutput, memory_sequence_length=encoderLengthList, normalize=('norm' in attentionMechanismName))
			if(beamSize > 1):
				beamAttentionMechanism = tf.contrib.seq2seq.BahdanauAttention(layerSize, memory=tiledMemory, memory_sequence_length=tiledLength, normalize=('norm' in attentionMechanismName))
		else:
			attentionMechanism = None
	# Dropout must be a placeholder/operation by now, as encoder will convert it
	assert isinstance(dropout, (tf.Operation, tf.Tensor)) and isinstance(correctResult, (tf.Operation, tf.Tensor))
	# Decoder construction here
	decoderCells = createRNNCell(cellType, layerSize, layerDepth, forgetBias, dropout=dropout, name=prefix+'_rnn')
	if(attentionMechanism is not None):
		if(beamSize > 1):
			# create a beam cell wrapped by a batched mechanism first, before resubtituing
			tiledState = tf.contrib.seq2seq.tile_batch(encoderState, beamSize)
			beamDecoderCells = tf.contrib.seq2seq.AttentionWrapper(decoderCells, beamAttentionMechanism, attention_layer_size=layerSize, initial_cell_state=tiledState, name=prefix + "_attention_wrapper")
		# set to reuse the beam if available
		with tf.variable_scope(tf.get_variable_scope(), reuse=(True if beamSize > 1 else tf.AUTO_REUSE)):
			decoderCells = tf.contrib.seq2seq.AttentionWrapper(decoderCells, attentionMechanism, attention_layer_size=layerSize, initial_cell_state=encoderState, name=prefix + "_attention_wrapper")
	# conversion layer to convert from the hiddenLayers size into vector size if they have a mismatch
	# Helper for training using the output taken from the encoder outside
	if('samplingVariable' not in settingDict):
		trainHelper = tf.contrib.seq2seq.TrainingHelper(decoderTrainingInput, correctResultLen)
	else:
		trainHelper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoderTrainingInput, correctResultLen, outputEmbedding, settingDict['samplingVariable'])
	# Projection layer
	projectionLayer = tf.layers.Dense(decoderOutputSize, name=prefix+'_projection') if(layerSize != decoderOutputSize) else None
	# create the initial state out by cloning
	if(attentionMechanism is None):
		initialState = encoderState
	else:
		# if using the AttentionWrapper, the confusingly named zero_state is the correct initial state since it record the inner initial_cell_state in it when we were creating the AttentionMechanism
		initialState = decoderCells.zero_state(dtype=tf.float32, batch_size=batchSize)
	trainDecoder = tf.contrib.seq2seq.BasicDecoder(decoderCells, trainHelper, initialState, output_layer=projectionLayer)
	# Helper for feeding the output of the current timespan for the inference mode
	if(beamSize > 1):
		# use beam search decoder, which will need specific kinds of input and a remade decoder cell
		beamInitialState = beamDecoderCells.zero_state(dtype=tf.float32, batch_size=batchSize*beamSize)
		inferDecoder = tf.contrib.seq2seq.BeamSearchDecoder(beamDecoderCells, outputEmbedding, tf.fill([batchSize], startToken), endToken, beamInitialState, beamSize, output_layer=projectionLayer)
	else:
		inferHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(outputEmbedding, tf.fill([batchSize], startToken), endToken)
		inferDecoder = tf.contrib.seq2seq.BasicDecoder(decoderCells, inferHelper, initialState, output_layer=projectionLayer)
	#else:
	#	initialState = initialState.clone(cell_state=encoderState)
	# depending on the mode being training or infer, use either Helper as fit
	# Another bunch of stuff I don't understand. Apparently the outputs and state are being created automatically. Yay.
	inferOutput, _, _ = tf.contrib.seq2seq.dynamic_decode(inferDecoder, maximum_iterations=maximumDecoderLength)
	if(beamSize > 1):
		# we are currently trimming the beam_width dimension away and keep the one with the best probs
		# the inferIds is [batch, tgt_len, beam] dimension. What the hell.
		inferIds = tf.squeeze(inferOutput.predicted_ids[:, :, :1], axis=[-1])
	else:
		inferIds = inferOutput.sample_id
	trainOutput, _, _ = tf.contrib.seq2seq.dynamic_decode(trainDecoder)
	trainLogits = trainOutput.rnn_output
	# inferLogits = inferOutput.rnn_output
	'''if(decoderOutputSize != layerSize):
		trainLogits = tf.layers.dense(trainLogits, decoderOutputSize, name='shared_projection', reuse=None)
		inferLogits = tf.layers.dense(trainLogits, decoderOutputSize, name='shared_projection', reuse=True)'''
	# sample_id is the argmax of the output. It can be used during beam searches and training
	# sample_id = trainOutput.sample_id
	lossOp, crossent = createSoftmaxDecoderLossOperation(trainLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength)
	# secondaryLossOp, secondaryCrossent = createSoftmaxDecoderLossOperation(inferLogits, correctResult, correctResultLen, batchSize, maximumDecoderLength)
	return trainLogits, lossOp, (trainOutput.sample_id, inferIds), crossent
	
def createOptimizer(settingDict):
	assert all(key in settingDict for key in ['mode', 'loss'])
	loss = settingDict['loss']
	mode = settingDict['mode']
	if(mode.lower() == 'sgd'):
		# optimizer = 
		trainingRate = settingDict['trainingRate']
		if(isinstance(trainingRate, float)):
			trainingRate = tf.constant(trainingRate, dtype=tf.float32, name='training_rate_default')
		# warmUp and decay scheme. Customize later
		if('globalSteps' in settingDict):
			globalSteps = settingDict['globalSteps']
			assert isinstance(globalSteps, tf.Variable)
		if('warmupTraining' in settingDict and 'globalSteps' in locals()):
			warmupStep, warmupThreshold = settingDict['warmupTraining']
			warmupFactor = tf.exp(-2 / warmupStep)
			inverseDecay = warmupFactor**(tf.to_float(warmupStep - globalSteps))
			trainingRate = tf.cond(globalSteps < warmupThreshold ,
									true_fn=lambda: inverseDecay * trainingRate,
									false_fn=lambda: trainingRate)
		if('decayTraining' in settingDict and 'globalSteps' in locals()):
			decayStep, decayThreshold, decayFactor = settingDict['decayTraining']
			staircase = not settingDict.get('disableStaircase', False)
			# warmupFactor = tf.exp(-2 / decayStep)
			# decayFactor = warmupFactor**(tf.to_float(warmupStep - globalSteps))
			trainingRate = tf.cond(globalSteps >= decayThreshold,
									true_fn=lambda: tf.train.exponential_decay(trainingRate, tf.maximum(globalSteps - decayThreshold, 0), decayStep, decayFactor, staircase=False),
									false_fn=lambda: trainingRate)
		
		return tf.train.GradientDescentOptimizer(trainingRate)
	elif(mode.lower() == 'adam'):
		if('trainingRate' not in settingDict):
			return tf.train.AdamOptimizer()
		else:
			trainingRate = settingDict['trainingRate']
			return tf.train.AdamOptimizer(trainingRate)
	else:
		raise Exception("Optimizer not specified.")
	
def configureGradientOptions(optimizer, settingDict):
	assert all(key in settingDict for key in ['colocateGradient', 'clipGradient', 'globalSteps', 'loss'])
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
	globalSteps = settingDict['globalSteps']
	return optimizer.apply_gradients(zippedGradientList, global_step=globalSteps)
	
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
	
def createSoftmaxDecoderLossOperation(logits, correctIds, sequenceLengthList, batchSize, maxUnrolling):
	# logits are undoing softmax, compare it with correctIds. the correctIds will converted to onehot upon use
	# raise the values in logits to prevent 0. May work or may not
	# logits = tf.clip_by_value(logits, 1e-10, 1e10)
	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=correctIds, logits=logits)
	# subtract = tf.reduce_mean(tf.square(tf.subtract(correctResult, logits)), axis=2)
	# mask to only calculate loss on the length of the sequence, not the padding
	target_weights = tf.sequence_mask(sequenceLengthList, maxUnrolling, dtype=tf.float32)
	# May not be the most efficient opperation, but I digress. Split the loss by the length of the correct sentence
	# should not use this one, as it reduce the target_weights far too low
#	target_weights = tf.transpose(tf.transpose(target_weights) / tf.to_float(sequenceLengthList))
	# the loss function being the reduce mean of the entire batch
	loss = tf.reduce_sum(tf.multiply(crossent, target_weights, name="crossent")) / tf.to_float(batchSize)
	return loss, target_weights
	
def createRNNCell(cellType, layerSize, layerDepth, dropout=None, name='RNN'):
	layers = []
	for i in range(layerDepth):
		cell = cellType(layerSize, name=name)
		if(dropout is not None):
			cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=dropout)
		layers.append(cell)
		
	return tf.contrib.rnn.MultiRNNCell(layers) if layerDepth > 1 else layers[0]
	
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

def createEmbeddingSessionCBOW(dictSize, embeddingSize, inputSize=1, trainingRate=1.0, existedSession=None, useNceLosses=True, numSample=9):
	# if use nce, create a lookup matrix and use default nce_loss function from tensorflow
	training_inputs = tf.placeholder(tf.int32, shape=[None, inputSize])
	training_outputs = tf.placeholder(tf.int32, shape=[None])
	embeddings = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize)), dtype=tf.float32, name="E")
	embed = tf.nn.embedding_lookup(embeddings, training_inputs)
	# tensor is being properly squashed from [batch_size, embeddingSize, inputSize] into [batch_size, embeddingSize*inputSize]
	#embed = tf.div(tf.reduce_sum(embed, 1), inputSize)
	embed = tf.reshape(embed, [-1, embeddingSize*inputSize])

	# Construct the variables for the NCE loss
	nce_weights = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize*inputSize)), dtype=tf.float32, name="EW")
	nce_biases = tf.Variable(tf.zeros([dictSize]))

	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each
	# time we evaluate the loss.
	extended_outputs = tf.expand_dims(training_outputs, -1)
	loss = tf.reduce_mean(tf.clip_by_value(
		tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=extended_outputs, inputs=embed, num_sampled=numSample, num_classes=dictSize), 1e-10, 100))
	global_step = tf.train.get_or_create_global_step()
	train_op = tf.contrib.layers.optimize_loss(loss, global_step, trainingRate, "SGD", clip_gradients=5.0, name="train_op")
	
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
	normalized_embeddings = embeddings / norm
	
	return sess, [train_op, loss], training_inputs, training_outputs, (embeddings, normalized_embeddings)
	
def createEmbeddingSession(dictSize, embeddingSize, inputSize=1, trainingRate=1.0, existedSession=None, useNceLosses=True, numSample=9):
	# if use nce, create a lookup matrix and use default nce_loss function from tensorflow
	training_inputs = tf.placeholder(tf.int32, shape=[None, inputSize])
	training_outputs = tf.placeholder(tf.int32, shape=[None])
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
	extended_outputs = tf.expand_dims(training_outputs, -1)
	loss = tf.reduce_mean(tf.clip_by_value(
		tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=extended_outputs, inputs=embed, num_sampled=numSample, num_classes=dictSize), 1e-10, 100))
	global_step = tf.train.get_or_create_global_step()
	train_op = tf.contrib.layers.optimize_loss(loss, global_step, trainingRate, "SGD", clip_gradients=5.0, name="train_op")
	
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	
	return sess, [train_op, loss], training_inputs, training_outputs, (embeddings, normalized_embeddings)
	
def createEnlargeEmbeddingSession(originalWordVectors, dictSize, embeddingSize, inputSize=1, trainingRate=1.0, existedSession=None, useNceLosses=True, numSample=9, isCBOW=False):
	original_dict_size = tf.convert_to_tensor(len(originalWordVectors), dtype=tf.int32)
	# if use nce, create a lookup matrix and use default nce_loss function from tensorflow
	training_inputs = tf.placeholder(tf.int32, shape=[None, inputSize])
	training_outputs = tf.placeholder(tf.int32, shape=[None])
	new_embeddings = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize)), dtype=tf.float32, name="E")
	original_embeddings = tf.constant(originalWordVectors, name="O")
	embeddings = tf.concat([original_embeddings, new_embeddings], axis=0)
	embed = tf.nn.embedding_lookup(embeddings, training_inputs)
	# stopgap - tensor is being sum together
	#embed = tf.div(tf.reduce_sum(embed, 1), inputSize)
	if(isCBOW):
		cbow_layer = tf.layers.Dense(embeddingSize, dtype=embed.dtype, name="C-L")
		embed = cbow_layer(embed)
	else:
		embed = tf.reduce_mean(embed, 1)

	# Construct the variables for the NCE loss
	nce_weights = tf.Variable(initial_value=createRandomArray((dictSize, embeddingSize)), dtype=tf.float32, name="EW")
	nce_biases = tf.Variable(tf.zeros([dictSize]), dtype=tf.float32, name="EB")

	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each
	# time we evaluate the loss.
	extended_outputs = tf.expand_dims(training_outputs, -1)
	raw_loss = tf.clip_by_value(
		tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=extended_outputs, inputs=embed, num_sampled=numSample, num_classes=dictSize), 1e-10, 100, name="L-C")
	# mask the loss so the original embedding does not get gradient (all constants)
	inputs_is_valid = tf.reduce_any(tf.math.less(training_inputs, original_dict_size), axis=-1, name="ML")
	masked_loss = tf.where(inputs_is_valid, x=raw_loss, y=0, name="L-M")
	global_step = tf.train.get_or_create_global_step()
	train_op = tf.contrib.layers.optimize_loss(loss, global_step, trainingRate, "SGD", clip_gradients=5.0, name="train_op")
	
	if(existedSession is None):
		sess = tf.Session()
	else:
		sess = existedSession
	
	norm = tf.sqrt(tf.reduce_sum(tf.square(new_embeddings), 1, keep_dims=True))
	normalized_new_embeddings = new_embeddings / norm
	
	return sess, [train_op, loss], training_inputs, training_outputs, (new_embeddings, normalized_new_embeddings)

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

def saveToPath(session, path, global_step=None):
	# add the saver into the session since it manipulate the checkpoint file
	saver = tf.train.Saver() if not hasattr(session, "saver") else session.saver
	saver.save(session, path, global_step=global_step)
	print("Saved data to location {}, global step value {}".format(path, global_step))
	session.saver = saver

def loadFromPath(session, path, checkpoint=None, debug=False):
	saver = tf.train.Saver() if not hasattr(session, "saver") else session.saver
	try:
		if(checkpoint):
			path = path + "-" + str(checkpoint)
			latest_path = path
		else:
			directory, file_name = os.path.split(path)
			latest_path = tf.train.latest_checkpoint(directory, latest_filename="checkpoint")
			if(latest_path == None):
				print("Latest checkpoint not found, should abort to skip")
			else:
				path = latest_path
		saver.restore(session, path)
		if(debug and (checkpoint is not None or latest_path is not None)):
			# only initiate with a verified checkpoint
			verifySessionLoadSuccess(session, latest_path)
			session.latest_path = latest_path
		print("Load data from location {} completed.".format(path))
#		loaded = True
	except tf.errors.NotFoundError:
		print("Skip the load process @ path {} due to file not existing.".format(path))
#		loaded = False
	except ValueError:
		print("Path {} not a valid checkpoint. Ignore and continue..".format(path))
#		loaded = False
#	if(not loaded):
#		print("Session not loaded, perform initialization by default")
#		session.run([tf.global_variables_initializer()])
	session.saver = saver
	return session

def loadCheckpointVariables(checkpoint_path, tensor_names):
	"""Copy from inspect_checkpoint tool"""
#	# import the needed tool
#	from tensorflow.python.tools import inspect_checkpoint
	reader = tf.train.NewCheckpointReader(checkpoint_path)
#	print(reader.debug_string())
#	checkpoint_namelist = reader.get_variable_to_shape_map()
	checkpoint_values = [reader.get_tensor(name) for name in tensor_names]
	return checkpoint_values

def verifySessionLoadSuccess(session, checkpoint_path, suppressError=True):
	"""Verify that all variables in the checkpoint is loaded into the session
		Args:
			session: the session to be checked
			checkpoint_path: the checkpoint path to be inspected and compare
			suppressError: if False, will raise Error on wrong load; if True, print out the result of comparison and continue
		Raise:
			a general Exception if encounter a wrong load and suppressError flag set to False
	"""
#	if(args.debug):
#		print("All trainable variables in session: {}".format([v.name for v in tf.trainable_variables()]))
	# load all variables in as numpy
	all_trainables = tf.trainable_variables()
	# names are without :0
	all_names = [v.name.replace(":0", "") for v in tf.trainable_variables()]
	session_values = session.run(all_trainables)
	checkpoint_values = loadCheckpointVariables(checkpoint_path, all_names)
	print("Begin checkpoint verification...")
	for name, session_val, checkpoint_val in zip(all_names, session_values, checkpoint_values):
		if(not np.array_equal(session_val, checkpoint_val)):
			if(not suppressError):
				raise Exception("Tensor with name {:s} load failed!\nSession is {}, Checkpoint is {}".format(name, session_val, checkpoint_val))
			else:
				print("ERROR! Tensor with name {:s} load failed!\nSession is {}, Checkpoint is {}".format(name, session_val, checkpoint_val))
		else:
			if(suppressError):
				print("Tensor {:s} load successfully".format(name))

def exportMetaGraph(file_stream, checkpoint_path=None):
	"""Load the meta graph at checkpoint_path and export it to a file
		Args:
			file_stream: the IOBuffer to feed the graph data into
			checkpoint_path: if true, load the meta graph into the running session
				WARNING! This operation will overwrite current default graph. Do not use during actual running operation
	"""
	if(checkpoint_path):
		saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
	all_graph_ops = tf.get_default_graph().get_operations()
	for item in all_graph_ops:
		file_stream.write(str(item))
	return file_stream

def _createTrashValue(dtype):
	if(dtype == tf.string):
		return "trash trash trash"
	elif(dtype.is_integer):
		return 0
	elif(dtype.is_floating):
		return 0.0
	elif(dtype.is_bool):
		return False
	else:
		raise ValueError("Dtype {} unsupported".format(dtype))

def _createTrashShapedValue(shape, dtype):
#	print("Creating trash {} with shape {}".format(dtype, shape))
	trash_value = _createTrashValue(dtype)
	# shapify
	for dim in reversed(shape):
		trash_value = [trash_value] * dim
	return np.array(trash_value)

def trySaveSessionValues(session, feed_dict=None):
	"""Save all session tensor state when feeded feed_dict|trash values """
	graph = tf.get_default_graph()
	if(feed_dict is None):
		print("Generate trash data..")
		placeholders = [x for x in graph.get_operations() if x.type == "Placeholder"]
		placeholder_tensors = [graph.get_tensor_by_name(x.name + ":0") for x in placeholders]
		placeholder_shapes = [x.get_shape().as_list() for x in placeholder_tensors]
		trash_shapes = [[dim if dim is not None else 1 for dim in shape] for shape in placeholder_shapes]
		trash_values = [_createTrashShapedValue(shape, tensor.dtype) for shape, tensor in zip(trash_shapes, placeholder_tensors)]
#		print(trash_values)
		feed_dict = {tensor:value for tensor, value in zip(placeholder_tensors, trash_values)}
	# only take fetchable (branched condition) and non-save non-training tensor
	all_tensors = [t for op in graph.get_operations() for t in op.values() if graph.is_fetchable(t)]
	exclusions = ("while","save", "train", "Gradient", "Initializer", "Assign")
	all_tensors = [t for t in all_tensors if all((ex not in t.name for ex in exclusions))]
#	print([t.name for t in all_tensors])
	all_tensor_values = session.run(all_tensors, feed_dict=feed_dict)
#	all_tensor_values = [val.tolist() if isinstance(val, np.ndarray) 
#											else str(val) if isinstance(val, (bytes, bytearray))
#											else val 
#														for val in all_tensor_values]
	return {t.name:v for t, v in zip(all_tensors, all_tensor_values)}
