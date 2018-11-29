import tensorflow as tf
import ffBuilder as builder

def buildSessionBidirectionalAttention(table_words, table_vectors, table_default_idx, cell_size=512, debug=False, gpu_allow_growth=True, optimizer = "SGD", learning_rate=1.0):
	"""Build the session using a bidirectional encoder and attention structure
		Args:
			table_words: list, words loaded from the embedding reader
			table_vectors: numpy array 2d, vectors correlating with the words
			table_default_idx: int, the index of the <unk> token
			cell_size: the size of the encoder cell
			debug: if True, add several tf.Print to process
			gpu_allow_growth: if set True, the tensorflow will only expand to the size it needs
		Returns:
			dict of
				session: finished tf.Session with the graph initialized
				placeholders: the placeholders used to either train or evaluate. inputs(str), outputs(float 0.0-1.0), dropout(float 0.0-1.0)
				training ops: the loss and train operation for training mode. tensor (float) and op
				predictions: the predictions in inference. tensor(float)
	"""
	# initialize the session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = gpu_allow_growth
	session = tf.Session(config=config)
	# anchor the embedding as constants
	embeddings_tensor = tf.constant(table_vectors, dtype=tf.float32, name="embedding_tensor")
	# create the word-to-id table
	word_indices, word_strings = zip(*enumerate(table_words))
	word_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(word_strings, word_indices), table_default_idx, name="word_to_id_table")
	# create the input placeholder, tokenize and feed through table
	input_placeholder = tf.placeholder(tf.string, shape=(None, ), name="batch_input")
	input_tokenized_sparse = tf.string_split(input_placeholder)
	input_tokenized_dense = tf.sparse_tensor_to_dense(input_tokenized_sparse, name="tokenized_input", default_value="<\s>")
	# use the input string in sparse tensor form to create the input_length vector
	sparse_indices = input_tokenized_sparse.indices
	input_length_sparse = tf.SparseTensor(sparse_indices, tf.squeeze(sparse_indices[:, 1:], axis=[1]), dense_shape=input_tokenized_sparse.dense_shape)
	input_length_dense = tf.sparse_tensor_to_dense(input_length_sparse, default_value=-1)
	input_length = tf.reduce_max(input_length_dense, axis=1)
	# lookup the ids by the table
	input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
	# create embedding and lookup the indices
	inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
	# RESULTS
	result_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="batch_result")
	result_sigmoid = tf.minimum(tf.cast(result_placeholder, tf.float32) - 1.0, 4.0) / 4.0
	# MISC
	dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
	# run bidirectional RNN 
	outputs, last_state = builder.createEncoderClean(inputs, cell_size=cell_size, created_dropout=dropout_placeholder)
	attention_base = tf.get_variable("attention_base", shape=[1, cell_size * 2, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
	batch_size = tf.shape(outputs)[0]
	attention_batch = tf.tile(attention_base, [batch_size, 1, 1])
	# compare directly, and mask
	# matrix multiply: [batch, length, num_units] * [batch, num_units, 1] = [batch, length, 1]
	attention_unmasked = tf.matmul(outputs, attention_batch)
	# masking log values, so select -inf if false
	mask_choice = tf.sequence_mask(input_length, maxlen=tf.shape(outputs)[1])
	mask_values = tf.fill(tf.shape(attention_unmasked), tf.float32.min)
	mask_choice = tf.expand_dims(mask_choice, axis=-1)
	# DEBUG
	if(debug):
		mask_choice = tf.Print(mask_choice, [tf.shape(input_length), tf.shape(mask_choice), tf.shape(attention_unmasked)])
	attention_masked = tf.where(mask_choice, attention_unmasked, mask_values)
	attention = tf.nn.softmax(attention_masked, name="attention")
	# compute context
	context_raw = tf.multiply(attention, outputs)
	context = tf.reduce_sum(context_raw, axis=1, name="context")
	# feed this context through an activation layer to get result
	predictions_raw = tf.layers.dense(context, 1, name="prediction_raw")
	predictions_raw = tf.squeeze(predictions_raw, axis=[-1])
	# compute loss from predictions to true (result)
	entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_sigmoid, logits=predictions_raw, name="cross_entropy")
	loss = tf.reduce_mean(entropy, name="loss")
	predictions = tf.nn.sigmoid(predictions_raw, name="prediction")
	if(debug):
		loss = tf.Print(loss, [result_sigmoid, predictions_raw, entropy, predictions])
	# create train_op
	global_step = tf.train.get_or_create_global_step()
	train_op = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, clip_gradients=1.0, name="train_op")
	# initialize
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	return {
		"session": session, 
		"placeholders": (input_placeholder, result_placeholder, dropout_placeholder), 
		"training_ops":(loss, train_op), 
		"predictions": predictions
	}

def buildSessionBidirectionalAttentionExtended(table_words, table_vectors, table_default_idx, cell_size=512, debug=False, gpu_allow_growth=True, optimizer = "SGD", learning_rate=1.0):
	"""An extended version from the buildSessionBidirectionalAttention, but instead of 1 there is now 3 prediction values: positivity, intensity and certainty, basing on the data of the comment.
		Args:
			table_words: list, words loaded from the embedding reader
			table_vectors: numpy array 2d, vectors correlating with the words
			table_default_idx: int, the index of the <unk> token
			cell_size: the size of the encoder cell
			debug: if True, add several tf.Print to process
			gpu_allow_growth: if set True, the tensorflow will only expand to the size it needs
		Returns:
			dict of
				session: finished tf.Session with the graph initialized
				placeholders: the placeholders used to either train or evaluate. inputs(str), outputs(float 0.0-1.0), dropout(float 0.0-1.0), comment_sure(float, 0.0-1.0)
				training ops: the loss and train operation for training mode. tensor (float) and op
				predictions: a tuple of:
					positivity: comment's idea (positive/negative)
					intensity: comment's strength (neutral/intense)
					certainty: how sure the session is of the prediction, use to guess comment_sure
	"""
	# initialize the session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = gpu_allow_growth
	session = tf.Session(config=config)
	# anchor the embedding as constants
	embeddings_tensor = tf.constant(table_vectors, dtype=tf.float32, name="embedding_tensor")
	# create the word-to-id table
	word_indices, word_strings = zip(*enumerate(table_words))
	word_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(word_strings, word_indices), table_default_idx, name="word_to_id_table")
	# create the input placeholder, tokenize and feed through table
	input_placeholder = tf.placeholder(tf.string, shape=(None, ), name="batch_input")
	input_tokenized_sparse = tf.string_split(input_placeholder)
	input_tokenized_dense = tf.sparse_tensor_to_dense(input_tokenized_sparse, name="tokenized_input", default_value="<\s>")
	# use the input string in sparse tensor form to create the input_length vector
	sparse_indices = input_tokenized_sparse.indices
	input_length_sparse = tf.SparseTensor(sparse_indices, tf.squeeze(sparse_indices[:, 1:], axis=[1]), dense_shape=input_tokenized_sparse.dense_shape)
	input_length_dense = tf.sparse_tensor_to_dense(input_length_sparse, default_value=-1)
	input_length = tf.reduce_max(input_length_dense, axis=1)
	# lookup the ids by the table
	input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
	# create embedding and lookup the indices
	inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
	# RESULTS
	result_placeholder = tf.placeholder(tf.int32, shape=(None, 2), name="batch_result")
	result_score_placeholder = tf.squueze(result_placeholder[:, :1], axis=[-1], name="batch_score")
	result_certainty_placeholder = tf.squueze(result_placeholder[: 1:2], axis=[-1], name="batch_certainty")
	# MISC
	dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
	# run bidirectional RNN 
	outputs, last_state = builder.createEncoderClean(inputs, cell_size=cell_size, created_dropout=dropout_placeholder)
	# the predictions
	prediction_intensity = createEncoderAttention(cell_size * 2, outputs, input_length, scope="intensity")
	prediction_positivity = createEncoderAttention(cell_size * 2, outputs, input_length, scope="positivity")
	prediction_certainty = createEncoderAttention(cell_size * 2, outputs, input_length, scope="certainty")
	# the correct values
	result_intensity =  tf.abs(tf.cast(result_score_placeholder, tf.float32) - 3.0) / 2.0
	result_positivity = tf.where(result_score_placeholder > 3, 1, 0)
	result_certainty = result_certainty_placeholder
	# the entropy
	entropy_intensity = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_intensity, logits=prediction_intensity)
	entropy_positivity_unmasked = tf.sigmoid_cross_entropy_with_logits(labels=result_positivity, logits=prediction_positivity)
	# for positivity, do not propel through those with neutral result (3)
	entropy_positivity = tf.where(result_placeholder == 3, 0, entropy_positivity)
	entropy_certainty = tf.sigmoid_cross_entropy_with_logits(labels=result_certainty, logits=prediction_certainty)
	# stack the entropy together as loss
	loss = entropy_intensity + entropy_positivity + entropy_certainty
	loss = tf.reduce_mean(loss, name="loss")
	# create train_op
	global_step = tf.train.get_or_create_global_step()
	train_op = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, clip_gradients=1.0, name="train_op")
	# initialize
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	return {
		"session": session, 
		"placeholders": (input_placeholder, result_placeholder, dropout_placeholder), 
		"training_ops":(loss, train_op), 
		"predictions": (prediction_intensity, prediction_positivity, prediction_certainty)
	}

def createEncoderAttention(attention_cell_size, encoder_outputs, encoder_output_lengths, scope="default"):
	"""Apply attention on top of the encoder outputs
		Args:
			attention_cell_size: int, must match encoder_outputs last dim (num_units)
			encoder_outputs: size [batch, length, num_units]
			encoder_length: size [batch]
		Returns:
			The unscaled prediction logits [batch]
	"""
	with tf.variable_scope(scope):
		attention_base = tf.get_variable("attention_base", shape=[1, attention_cell_size * 2, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
		batch_size = tf.shape(outputs)[0]
		attention_batch = tf.tile(attention_base, [batch_size, 1, 1])
		# compare directly, and mask
		# matrix multiply: [batch, length, num_units] * [batch, num_units, 1] = [batch, length, 1]
		attention_unmasked = tf.matmul(outputs, attention_batch)
		# masking log values, so select -inf if false
		mask_choice = tf.sequence_mask(input_length, maxlen=tf.shape(outputs)[1])
		mask_values = tf.fill(tf.shape(attention_unmasked), tf.float32.min)
		mask_choice = tf.expand_dims(mask_choice, axis=-1)
		# apply the mask and get the attention
		attention_masked = tf.where(mask_choice, attention_unmasked, mask_values)
		attention = tf.nn.softmax(attention_masked, name="attention")
		# compute context
		context_raw = tf.multiply(attention, outputs)
		context = tf.reduce_sum(context_raw, axis=1, name="context")
		# feed this context through an activation layer to get result
		predictions_raw = tf.layers.dense(context, 1)
		predictions_raw = tf.squeeze(predictions_raw, axis=[-1], name="prediction_raw")
	return predictions_raw
