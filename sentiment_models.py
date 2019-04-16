import tensorflow as tf
import keras
import ffBuilder as builder
import numpy as np
import os, time, io
import abc
from sentiment_eval import createRatingMaps, createHeatmapImage
from random import shuffle

_DEFAULT_OPTIMIZER = {
	"adam": {"learning_rate": 0.0001, "clip_gradients": 5.0},
	"sgd": {"learning_rate": 1.0, "learning_rate_decay_fn": lambda lr, gs: tf.train.exponential_decay(lr, gs, 10000, 0.5), "clip_gradients": 5.0}
}

class SentimentModel:
	def __init__(self):
		raise NotImplementedError("Base abstract model, cannot initialize")

	@abc.abstractmethod
	def buildSession(self, *args, **kwargs):
		raise NotImplementedError("Base model, no session")
	
	@abc.abstractmethod
	def trainSession(self, dataset):
		raise NotImplementedError("Base model, no session")

	@abc.abstractmethod
	def evalSession(self, dataset, detailed=False):
		raise NotImplementedError("Base model, no session")

	@abc.abstractmethod
	def inferSession(self, data):
		raise NotImplementedError("Base model, no session")
	
	@abc.abstractmethod
	def exportSession(self):
		raise NotImplementedError("Base model, no session")

	@abc.abstractmethod
	def tune(self, dataset):
		raise NotImplementedError("Tuning not implemented")

	def saveSession(self, compare_func, model_idx=None):
		if(not compare_func):
			compare_func = self._default_compare_function
		better, new_stat = compare_func(self._prev_stat, self._current_stat)
#		print("Compare {} to {}, result {} {}".format(self._prev_stat, self._current_stat, better, new_stat))
		if(better):
			if(self._epoch_tensor):
				session = self._session_dictionary["session"]
				self._epoch_tensor.load(self._epoch, session)
				model_idx = model_idx if model_idx is not None else self._epoch
			builder.saveToPath(session, self._save_path, global_step=model_idx)
			return new_stat
		else:
			return self._prev_stat

	def loadSession(self, checkpoint=None):
		session = self._session_dictionary["session"]
		builder.loadFromPath(session, self._save_path, checkpoint=checkpoint, debug=self._debug)
		if(self._epoch_tensor):
			self._epoch = session.run(self._epoch_tensor)

	def _default_compare_function(self, prev, curr):
		if(prev == None):
			return True, curr
		return any((a > b for a, b in zip(curr, prev))), tuple([a if a > b else b for a, b in zip(curr, prev)])

	def _exportSession(self, session, input_set, output_set, export_sub_folder=0, override=False):
		export_path = self._export_path
		# create builder
		model_export_path = os.path.join(export_path, str(export_sub_folder))
		if(override and os.path.isdir(model_export_path)):
			print("Exported model detected, override on. Removing..")
			import shutil
			shutil.rmtree(model_export_path)
		export_builder = tf.saved_model.builder.SavedModelBuilder(model_export_path)
		# load info
		input_set = {k:tf.saved_model.utils.build_tensor_info(v) for k, v in input_set.items()}
		output_set = {k:tf.saved_model.utils.build_tensor_info(v) for k, v in output_set.items()}
		# create signature to be recalled in serving model
		prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=input_set, outputs=output_set, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
		export_builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], signature_def_map={
			"serving_default": prediction_signature
		}, main_op=tf.tables_initializer(), strip_default_attrs=True)
		export_builder.save()
		print("Serve-able model exported at location {}, subfolder {}".format(export_path, export_sub_folder))

	def _buildLookupTable(self, words, table_vectors, trainable_words=None, default_word_idx=None, scope="embedding"):
		"""Build a lookup hash table
			Args:
				words: list of words with accompany vectors. list of str
				table_vectors: list of accompanying vectors. 2d of [num_words, vector_size]
				trainable_words: list of words without vectors, trainable. list of str
				default_word_idx: the index to point at when a word is not recognized
			Returns:
				a word_table object which is a HashTable lookup in tensorflow, and an embeddings_tensor to convert it to vector
		"""
		# anchor the embedding as constants if found
		with tf.variable_scope(scope):
			if(len(words) == 0):
	#			raise ValueError("Should not be in here, since even no pretrain must have unk anchored")
				print("No pretrained value, assume table_vectors contain the size of the created embedding, which is {:d}".format(table_vectors))
				assert isinstance(default_word_idx, str), "Default word must be str for untrained"
				embedding_size = table_vectors
				embeddings_tensor = tf.get_variable("trainable_words_tensor", shape=[len(trainable_words)+1, embedding_size], dtype=tf.float32)
				words = [default_word_idx] + trainable_words
				default_idx = 0
			else:
				assert isinstance(default_word_idx, int), "Default word must be int for trained"
				embeddings_tensor = tf.constant(table_vectors, dtype=tf.float32, name="embeddings_tensor")
				if(trainable_words and isinstance(trainable_words, list)):
					print("Additional word list detected, length {:d}".format(len(trainable_words)))
					trainable_addtional_tensor = tf.get_variable("trainable_words_tensor", shape=[len(trainable_words), np.shape(embeddings_tensor)[1]], dtype=tf.float32)
					# join the extra words
					embeddings_tensor = tf.concat([embeddings_tensor, trainable_addtional_tensor], axis=0)
					words.extend(trainable_words)
				default_idx = default_word_idx
		# create the word-to-id table
		word_indices, word_strings = zip(*enumerate(words))
		word_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(word_strings, word_indices), default_idx, name="word_to_id_table")
		return word_table, embeddings_tensor
	
	def _buildInputSection(self):
		"""Build an input placeholder that read a batch of strings and output tokenized words and length
			Returns:
				placeholder of (batchsize,) string, tensor of (batch_size, length) str (words), tensor of (batch_size,) int (correct_length)
		"""
		# create the input placeholder, tokenize and feed through table
		input_placeholder = tf.placeholder(tf.string, shape=(None, ), name="batch_input")
		input_tokenized_sparse = tf.string_split(input_placeholder)
		input_tokenized_dense = tf.sparse_tensor_to_dense(input_tokenized_sparse, name="tokenized_input", default_value="<\s>")
		# use the input string in sparse tensor form to create the input_length vector
		sparse_indices = input_tokenized_sparse.indices
		input_length_sparse = tf.SparseTensor(sparse_indices, tf.squeeze(sparse_indices[:, 1:], axis=[1]), dense_shape=input_tokenized_sparse.dense_shape)
		input_length_dense = tf.sparse_tensor_to_dense(input_length_sparse, default_value=-1)
		# input_length must have +1 since it is actually max index
		input_length = tf.reduce_max(input_length_dense, axis=1) + 1
		return input_placeholder, input_tokenized_dense, input_length

	def _buildTrainingInputSection(self):
		"""Build the necessary placeholders to compare the predictions to
			For current model, the placeholders read a rating (1-5) of int
			Return the placeholders and the sigmoids equivalent
		"""
		# RESULTS
		result_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="batch_result")
		result_sigmoid = tf.minimum(tf.cast(result_placeholder, tf.float32) - 1.0, 4.0) / 4.0
		return result_placeholder, result_sigmoid

	def _createTrainingOp(self, entropy, optimizer, result_score=None):
		"""Create the training ops basing on the values
			Args:
				entropy: float [batch_size], value to minimize
				optimizer: str, the optimizer to use
				result_score: float/int [batch_size], the correct score
			Returns:
				the training operation to be invoked during training
		"""
		if(self._loss_weight is not None):
			print("Use weighted loss")
			int_result_score = tf.to_int32(result_score)
			loss_weights = self._buildWeightedLoss(int_result_score)
			entropy = entropy * loss_weights
		loss = tf.reduce_mean(entropy, name="loss")
		global_step = tf.train.get_or_create_global_step()
		extra_kwargs = _DEFAULT_OPTIMIZER[optimizer.lower()]
		learning_rate = extra_kwargs.pop("learning_rate")
		return tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, name="train_op", **extra_kwargs), loss

	def _retrieveEpoch(self):
		"""Create a tensor to record the current epoch
			Returns:
				a tensor of type int
		"""
		self._epoch = 0
		self._epoch_tensor = tf.Variable(0, name="epoch")
		return self._epoch_tensor


class KerasRNNModel(SentimentModel):
	def __init__(self, save_path, export_path, batch_size=128, debug=False, shuffle_batch=True, loss_weight=None):
		if(loss_weight is not None):
			assert isinstance(loss_weight, dict)
			self._loss_weight = [ float(loss_weight[str(i+1)]) for i in range(5)]
		else:
			self._loss_weight = None
		self._save_path = save_path
		if(not os.path.exists(save_path)):
			print("Path not exist: {:s}. Create new.".format(save_path))
			os.mkdir(save_path)
		self._export_path = export_path
		self._batch_size = batch_size
		self._debug = debug
		self._shuffle_batch = shuffle_batch
		self._prev_stat = None
		self._attention_names = None
		self._max_sequence_length = 100

	def buildSession(self, table_words, table_vectors, table_default_idx, cell_size=512, additional_words=None, gpu_allow_growth=True, optimizer="SGD"):
		if(isinstance(table_default_idx, str)):
			assert table_default_idx not in additional_words
			additional_words.insert(0, table_default_idx)
			self._default_idx = 0
		else:
			self._default_idx = table_default_idx
		self._vocab = vocab = {word:idx for idx, word in enumerate(additional_words)}
		self._model = model = keras.Sequential()
		model.add(keras.layers.Embedding(len(additional_words), cell_size, input_length=self._max_sequence_length))
		model.add(keras.layers.LSTM(128))
		model.add(keras.layers.Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		model.summary()
		self._session_dictionary = {"session": model, "vocab": vocab}

	def _reformData(self, dataset):
		# remove the certainty and all neutral values
		features, labels, *_ = zip(* ((f, l) for f, l in dataset if int(l) != 3))
		vocab, default_idx = self._vocab, self._default_idx
		process_line = lambda line: [ vocab.get(word, default_idx) for word in line.split()]
		features = np.array([ process_line(line) for line in features ])
		# 0 for neg, 1 for pos, already without neutral
		labels = np.array([ int(int(l) > 3) for l in labels ])
		# pad the sentences and unlock the labels
		features = keras.preprocessing.sequence.pad_sequences(features, maxlen=self._max_sequence_length)
		labels = np.reshape(labels, [-1, 1])
		return features, labels


	def trainSession(self, training_dataset, eval_dataset, epoch, dropout=1.0):
		model = self._model
		train_features, train_labels = self._reformData(training_dataset)
#		train_features, train_labels = np.arange(500).reshape([2, 250]), np.array([[0.0], [0.5]])
		print(train_features.shape, train_labels.shape)
		eval_compatible_set = self._reformData(eval_dataset)
		model.fit(train_features, train_labels, validation_data=eval_compatible_set, batch_size=self._batch_size, epochs=epoch, callbacks=[keras.callbacks.ModelCheckpoint(os.path.join(self._save_path, "model.h5"), monitor='val_loss', save_best_only=True)]) 

	def evalSession(self, eval_dataset, detailed=False, alignment_sample=False, file_path=None, external_fn=None):
		eval_features, eval_labels = self._reformData(eval_dataset)
		eval_loss, eval_acc = self._model.evaluate(eval_features, eval_labels, batch_size=self._batch_size)
		print("Eval: loss {:.4f}, accuracy {:.2f}%".format(eval_loss, eval_acc * 100.0))
		return eval_loss, eval_acc

	def inferSession(self, dataset, external_fn=None):
		model = self._model
		return model.predict(dataset, batch_size=self._batch_size, verbose=1)

	def saveSession(self, compare_func, model_idx=None):
		self._model.save(os.path.join(self._save_path, "model.h5"))
	
	def loadSession(self, checkpoint=None):
		if(os.path.isfile(os.path.join(self._save_path, "model.h5"))):
			print("Found model in path, overriding..")
			del self._model
			self._model = keras.models.load_model(os.path.join(self._save_path, "model.h5"))
			self._model.summary()

	def tune(self, dataset, **kwargs):
		print("Currently tuning is disabled on all Keras models")

class KerasVDCNNModel(KerasRNNModel):
	def buildSession(self, table_words, table_vectors, table_default_idx, cell_size=512, additional_words=None, gpu_allow_growth=True, optimizer="SGD", convolution_size=128, additive=False):
		if(isinstance(table_default_idx, str)):
			assert table_default_idx not in additional_words
			additional_words.insert(0, table_default_idx)
			self._default_idx = 0
		else:
			self._default_idx = table_default_idx
		self._vocab = vocab = {word:idx for idx, word in enumerate(additional_words)}
		model_input = keras.layers.Input(shape=(100, ), dtype='int32')
		embedded = keras.layers.Embedding(len(additional_words), cell_size, input_length=self._max_sequence_length) (model_input)
		if(additive):
			embedded = keras.layers.Dropout(0.5) ( keras.layers.AdditiveLayer() (embedded) )
		# first convolution layer, 
		conv_ops = []
		for filter_size in range(2, 6):
			conv = keras.layers.Conv1D(convolution_size, filter_size, activation='relu') (embedded)
			pooling = keras.layers.MaxPool1D(5) (conv)
			conv_ops.append(pooling)
		all_conv_filter = keras.layers.Concatenate(axis=1) (conv_ops)
		conv_1 = keras.layers.BatchNormalization() (all_conv_filter)
		# second convolution layer, apply atop conv_1 and have the same size
		conv_2 = keras.layers.BatchNormalization() ( keras.layers.Conv1D(convolution_size, 5, padding="same") (conv_1))
		conv_2 = keras.layers.BatchNormalization() ( keras.layers.Conv1D(convolution_size, 5, padding="same") (conv_2))
		conv_2 = keras.layers.MaxPool1D() ( keras.layers.Add() ([conv_1, conv_2] ) )
		# third convolution layer, same as conv_2
		conv_3 = keras.layers.BatchNormalization() ( keras.layers.Conv1D(convolution_size, 5, padding="same") (conv_2))
		conv_3 = keras.layers.BatchNormalization() ( keras.layers.Conv1D(convolution_size, 5, padding="same") (conv_3))
		conv_3 = keras.layers.MaxPool1D() ( keras.layers.Add() ([conv_2, conv_3] ) )
		
		features_sparse = keras.layers.Flatten() (conv_3)
		features_dense = keras.layers.Dense(64, activation='relu') (features_sparse)
		features_dense = keras.layers.BatchNormalization() (features_dense)
		model_output = keras.layers.Dense(1, activation='softmax') (features_dense)
		self._model = model = keras.Model(inputs=model_input, outputs=model_output)
		model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
		model.summary()
		self._session_dictionary = {"session": model, "vocab": vocab}

class SentimentCNNPooledModel(SentimentModel):
	def __init__(self, save_path, export_path, batch_size=128, debug=False, shuffle_batch=True, loss_weight=None):
		self._save_path = save_path
		self._export_path = export_path
		self._batch_size = batch_size
		self._debug = debug
		self._shuffle_batch = shuffle_batch

	def buildSession(self, table_words, table_vectors, table_default_idx, cell_size=512, additional_words=None, gpu_allow_growth=True, optimizer="SGD"):
		"""Build the session using a bidirectional encoder and CNN's pooling
			Args:
				table_words: list, words loaded from the embedding reader
				table_vectors: numpy array 2d, vectors correlating with the words
				table_default_idx: int, the index of the <unk> token
				cell_size: the size of the encoder cell
				additional_words: a list of additional words to be added into table_words if exist. Their vectors will be randomized and trained unlike the table_vectors
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
		# create the lookup word table
		word_table, embeddings_tensor = self._buildLookupTable(table_words, table_vectors, trainable_words=additional_words, default_word_idx=table_default_idx)
		# create the inputs and lookup the ids by the table
		input_placeholder, input_tokenized_dense, input_length = self._buildInputSection()
		input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
		# lookup the indices by the embeddings
		inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
		# MISC
		result_placeholder, result_sigmoid = self._buildTrainingInputSection()
		dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
		# directly apply the CNN over the inputs
		predictions_raw, entropy = self._buildPredictionSection(inputs, input_length, tf.shape(inputs)[-1], result_placeholder)

		# compute loss from predictions to true (result)
		predictions = tf.nn.sigmoid(predictions_raw, name="prediction_sigmoid")
		predictions_rating = tf.cast(1.5 + predictions * 4.0, dtype=tf.int32, name="predictions")
		# create train_op
		train_op, loss = self._createTrainingOp(entropy, optimizer, result_score=result_placeholder)
		# initialize
		self._session_dictionary = {
			"session": session, 
			"placeholders": (input_placeholder, result_placeholder, dropout_placeholder), 
			"training_ops":(loss, train_op), 
			"predictions_raw": predictions,
			"predictions": predictions_rating
		}
		return self._session_dictionary

	def _buildCNN(self, inputs, cnn_config, dropout, dense_units_size=128):
		inputs = tf.expand_dims(inputs, axis=-1)
		layer_result = inputs
		# construct CNN using config
		for idx, item in enumerate(cnn_config):
			(conv2d_filter_size, conv2d_kernel_size), (pool_size, pool_stride) = item
			conv2d_layer = tf.layers.conv2d(layer_result, filters=conv2d_filter_size, kernel_size=conv2d_kernel_size, activation=tf.nn.relu, name="conv2d_layer_{:d}".format(idx))
			pool_layer = tf.layers.max_pooling2d(conv2d_layer, pool_size=pool_size, strides=pool_stride, name="pool_layer_{:d}".format(idx))
			layer_result = pool_layer
		# flatten and use dense layer
		flattened_result = tf.reshape(layer_result, [tf.shape(inputs)[0], -1])
		dense_result = tf.layers.dense(flattened_result, dense_units_size, name="dense_first_layer")
		# add dropout
		result_with_dropout = tf.layers.dropout(dense_result, rate=dropout)
		logits = tf.layers.dense(result_with_dropout, 1, name="logits")
		return logits

	def _buildPredictionSection(self, cnn_output, cnn_length, cnn_size, result_sigmoid):
		"""Build the prediction part on a CNN structure
			Args:
				cnn_output: the output of the cnn process, [batch_size, length, cnn_size]
				cnn_length: the length of the input, use to mask the result
				cnn_size: the size of the cnn process
				result_sigmoid: the correct result, converted to sigmoid (0-1)
			Returns:
				tuple of:
					predictions: the raw prediction, unscaled (-inf, inf), [batch_size]
					entropy: the entropy between the prediction and the result, [batch_size]
		"""
		predictions_raw, predictions_attention = createEncoderAttention(cnn_size, cnn_output, cnn_length)
		entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_sigmoid, logits=predictions_raw, name="entropy")
		predictions_attention = tf.transpose(predictions_attention, perm=[0, 2, 1])
		return predictions_raw, predictions_attention, entropy






class SentimentRNNAttention(SentimentModel):
	def __init__(self, save_path, export_path, batch_size=128, debug=False, shuffle_batch=True, loss_weight=None):
		if(loss_weight is not None):
			assert isinstance(loss_weight, dict)
			self._loss_weight = [ float(loss_weight[str(i+1)]) for i in range(5)]
		else:
			self._loss_weight = None
		self._save_path = save_path
		self._export_path = export_path
		self._batch_size = batch_size
		self._debug = debug
		self._shuffle_batch = shuffle_batch
		self._prev_stat = None
		self._attention_names = None

	def _buildWeightedLoss(self, result_tensor):
		"""This assume that self._loss_weight exist as a dict.
			Args:
				result_tensor: the tensor containing the correct rating. [batch_size] of int 1-5
			Returns:
				a correct tensor of [batch_size], containing the weights
		"""
		# the weight are inversely proportional to its percentage to the sum, thus it is sum / occ
		sum_loss_weight = sum(self._loss_weight)
		wl = [sum_loss_weight / occurence for occurence in self._loss_weight]
#		print("Use weight = {}".format(wl))
		weighted_loss = tf.constant(wl, name="weighted_loss")
		# create the selector
		# need a -1 since it ran from 1-5 while we want selector in range 0-4 per bracket
		selector = result_tensor - 1
		# select
		loss_weights = tf.gather(weighted_loss, selector, name="loss_weights")
		return loss_weights

	def _buildPredictionSection(self, rnn_output, rnn_length, rnn_size, result_sigmoid):
		"""Build the prediction part atop the result of the bidirectional RNN set
			Args:
				rnn_output: the output of the RNN process, [batch_size, length, rnn_size]
				rnn_length: the length of the input, use to mask the result
				rnn_size: the size of the RNN process
				result_sigmoid: the correct result, converted to sigmoid (0-1)
			Returns:
				tuple of:
					predictions: the raw prediction, unscaled (-inf, inf), [batch_size]
					attentions: the attention(s) used during the process, [batch_size, 1, length]
					entropy: the entropy between the prediction and the result, [batch_size]
		"""
		predictions_raw, predictions_attention = createEncoderAttention(rnn_size, rnn_output, rnn_length)
		entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_sigmoid, logits=predictions_raw, name="entropy")
		predictions_attention = tf.transpose(predictions_attention, perm=[0, 2, 1])
		return predictions_raw, predictions_attention, entropy

	def buildSession(self, table_words, table_vectors, table_default_idx, cell_size=512, additional_words=None, gpu_allow_growth=True, optimizer="SGD"):
		"""Build the session using a bidirectional encoder and attention structure
			Args:
				table_words: list, words loaded from the embedding reader
				table_vectors: numpy array 2d, vectors correlating with the words
				table_default_idx: int, the index of the <unk> token
				cell_size: the size of the encoder cell
				additional_words: a list of additional words to be added into table_words if exist. Their vectors will be randomized and trained unlike the table_vectors
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
		self._retrieveEpoch()
		# create the lookup word table
		word_table, embeddings_tensor = self._buildLookupTable(table_words, table_vectors, trainable_words=additional_words, default_word_idx=table_default_idx)
		# create the inputs and lookup the ids by the table
		input_placeholder, input_tokenized_dense, input_length = self._buildInputSection()
		input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
		# lookup the indices by the embeddings
		inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
		# MISC
		dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
		# run bidirectional RNN 
		outputs, last_state = builder.createEncoderClean(inputs, cell_size=cell_size, created_dropout=dropout_placeholder)
		result_placeholder, result_sigmoid = self._buildTrainingInputSection()
		predictions_raw, attention, entropy = self._buildPredictionSection(outputs, input_length, cell_size*2, result_sigmoid)
		# compute loss from predictions to true (result)
		predictions = tf.nn.sigmoid(predictions_raw, name="prediction_sigmoid")
		predictions_rating = tf.cast(1.5 + predictions * 4.0, dtype=tf.int32, name="predictions")
		# create train_op
		train_op, loss = self._createTrainingOp(entropy, optimizer, result_score=result_placeholder)
		# initialize
		self._session_dictionary = {
			"session": session, 
			"placeholders": (input_placeholder, result_placeholder, dropout_placeholder), 
			"training_ops":(loss, train_op), 
			"predictions_raw": predictions,
			"predictions": predictions_rating,
			"attentions": attention
		}
		return self._session_dictionary

	def trainSession(self, training_dataset, eval_dataset, epoch, dropout=1.0):
		"""Train the session on the training_dataset and evaluate eval_dataset if exist
			Args:
				training_dataset: the iterable containing (line, score) both in raw format (str)
				eval_dataset: the iterable containing (line, score) both in raw format (str)
				epoch: the number of runs through the entire dataset
				dropout: if specified, the dropout will be feed this value
		"""
		session_dictionary = self._session_dictionary
		# preparation
		session = session_dictionary["session"]
		input_pl, result_pl, dropout_pl = session_dictionary["placeholders"]
		loss_t, train_op = trainers = session_dictionary["training_ops"] 
#		predictions = session_dictionary["predictions"]
		training_dropout = dropout
		timer = time.time()
		
		for i in range(self._epoch, epoch):
			# start training
			batch_iters = batch(training_dataset, n=self._batch_size, shuffle_batch=self._shuffle_batch)
			for batch_idx, iter_batch in enumerate(batch_iters):
				# drop the later args to fit both data_processing mode
				inputs, correct_outputs, *_ = zip(*iter_batch)
				inputs = list(inputs)
				correct_outputs = [int(o) for o in correct_outputs]
				loss, _ = session.run(trainers, feed_dict={input_pl:inputs, result_pl:correct_outputs, dropout_pl:training_dropout})
				if(self._debug):
					print("Iter {:d}, batch {:d}, loss {:.4f}, time passed {:.2f}".format(i, batch_idx, loss, time.time() - timer))
			print("Epoch iter {:d} finished, time passed {:.2f}".format(i, time.time() - timer))
			# start eval if eval dataset available
			if(not eval_dataset):
				self._saveSession(None)
				continue
			self._epoch += 1
			self._current_stat = self.evalSession(eval_dataset, detailed=False)
			self._prev_stat = self.saveSession(self._default_compare_function, model_idx=(i+1))
		print("All training phase completed, time passed {:.2f}".format(time.time()-timer))
	
	def evalSession(self, eval_dataset, detailed=False, alignment_sample=False, file_path=None, external_fn=None):
		"""Evaluate the session on the eval_dataset
			Args:
				eval_dataset: the iterable containing (line, score) both in raw format (str)
				detailed: if true, print the trace of every line
				alignment_sample: if True, evaluate the result and print out the best and worst sample. Currently set to 10 samples each
				file_path: the path to print heatmap or alignment_sample. str
				external_fn: a callable that will change the ratings. Use to implement dictionary
			Returns:
				a tuple of correct_score, correct_positivity, and negative mean_difference to help during saveSession if not in alignment mode
		"""
		# rip the eval datasets out, and ignore the reliability rating if they exist
		lines, ratings, *_ = zip(*eval_dataset)
		ratings = (int(rat) for rat in ratings)
		scored_dataset = self._scoreDataset(lines, alignment_sample=alignment_sample, external_fn=external_fn)
		# what a disaster this is. adding correct to the formatted (lines, preds, ..)
		scored_dataset = zip(ratings, *zip(*scored_dataset))
		# the scored is (correct_score, line, predictions, ?alignment?) per each sample
		if(not alignment_sample):
			def print_line(correct_rating, sentence, pred_rating):
				if(detailed):
					print("({:s}) Prediction {:d}, correct {:d} ||| {:s}".format("X" if pred_rating==correct_rating else " ", pred_rating, correct_rating, sentence))
			def print_final(total, pos_score, rat_score, dev_score):
				correct_pos_str = "{:d}/{:d}({:.2f}%)".format(pos_score, total, float(pos_score) / float(total) * 100.0)
				correct_rat_str = "{:d}/{:d}({:.2f}%)".format(rat_score, total, float(rat_score) / float(total) * 100.0)
				mean_dev_str = "{:d}/{:d}(Average {:.2f})".format(dev_score, total, float(dev_score) / float(total))
				print("Eval result: Positivity {:s}, Rating {:s}, Mean deviation {:s}".format(correct_pos_str, correct_rat_str, mean_dev_str))
				return rat_score, pos_score, -dev_score
			scored_dataset = list(scored_dataset)
			result = self._showResult(scored_dataset, per_line_func=print_line, end_func=print_final)
			if(file_path):
				ratingMapPath = os.path.splitext(file_path)[0] + "_rating.png"
				with io.open(ratingMapPath, "wb") as ratingMapFile:
					ratingMapFile = createRatingMaps(ratingMapFile, scored_dataset, image_title="Result Heatmap")
				print("Rating map exported to {:s}".format(ratingMapPath))
			return result
		else:
			print("Entering alignment viewing mode")
			assert file_path is not None, "Must have file_path"
			file_base, extension = os.path.splitext(file_path)
			# list-ify to prevent sort from running them out
			scored_dataset = list(scored_dataset)
			# score base on same positivity and closeness in prediction score
			# sort go from smallest to greatest, so comparer is 0 if the same, 1 if slightly different but same pos, 2..5 if different pos
			comparer = lambda i1, i2: (0 if i1 == i2 or (i1 - 3) * (i2 - 3) > 0 else 1) + abs(i2 - i1)
			# slightly biased on sentence with length around 35
			best_samples = sorted(scored_dataset, key=lambda item: comparer(item[0], item[2])*100 + abs(len(item[1]) - 35))[:10]
			best_file_path = file_base + "_best.png"
			with io.open(best_file_path, "wb") as best_file:
				best_actual, best_sentences, best_predict, best_alignments = zip(*best_samples)
				best_score = zip(best_predict, best_actual)
				#print(np.shape(best_alignments))
				createHeatmapImage(best_file, best_sentences, best_alignments, best_score, image_title="Best Result", attention_names=self._attention_names)
			print("Exported best results to {}".format(best_file_path))

			worst_samples = sorted(scored_dataset, key=lambda item: comparer(item[0], item[2])*100 - abs(len(item[1]) - 35), reverse=True)[:10]
			worst_file_path = file_base + "_worst.png"
			with io.open(worst_file_path, "wb") as worst_file:
				worst_actual, worst_sentences, worst_predict, worst_alignments = zip(*worst_samples)
				worst_score = zip(worst_predict, worst_actual)
				#print(np.shape(worst_alignments))
				createHeatmapImage(worst_file, worst_sentences, worst_alignments, worst_score, image_title="Worst Result", attention_names=self._attention_names)
			print("Exported worst results to {}".format(worst_file_path))

	def _scoreDataset(self, dataset, alignment_sample=False, external_fn=None):
		"""Evaluate the session on the dataset
			Args:
				dataset: the iterable containing (line) in raw format (str)
				alignment_sample: if True, append alignments data
				external_fn: an external function to change the predictions
			Returns:
				tupled and iterable data containing prediction score
		"""
		# expand the dataset, run through session to obtains scores
		session = self._session_dictionary["session"]
		input_pl, _, _ = self._session_dictionary["placeholders"]
		inputs = list(dataset)
		batched_inputs = batch(inputs, n=self._batch_size, shuffle_batch=False)
		prediction_tensor = self._session_dictionary["predictions"]
		if(alignment_sample):
			alignment_tensor = self._session_dictionary["attentions"]
			predictions, alignments = [], []
			for line_batch in batched_inputs:
				# eval for each batch with alignments
				preds, aligns = session.run([prediction_tensor, alignment_tensor], feed_dict={input_pl:line_batch})
				predictions.extend(preds)
				alignments.extend(aligns)
			if(external_fn and callable(external_fn)):
				# run rectifier if exist
				print("Using internal rectifiers")
				inputs, predictions = external_fn(inputs, predictions)
			return list(zip(inputs, predictions, alignments))
		else:
			predictions = []
			for line_batch in batched_inputs:
				# eval for each batch without alignments
				preds = session.run(prediction_tensor, feed_dict={input_pl:line_batch})
				predictions.extend(preds)
			if(external_fn and callable(external_fn)):
				print("Using internal rectifiers")
				# run rectifier if exist
				inputs, predictions = external_fn(inputs, predictions)
			return zip(inputs, predictions)

	def inferSession(self, dataset, external_fn=None):
		"""Infer data's score
			Args:
				dataset: iterable of (line)
				external_fn: a function that can change the predictions. a fn(inputs, predictions)
			Returns:
				the predictions for each line, int in range 1-5
		"""
		data = self._scoreDataset(dataset, external_fn=external_fn)
		lines, predictions = zip(*data)
		return predictions

	def _showResult(self, bundled_data_and_scores, per_line_func=None, end_func=None):
		"""Evaluation function: Calculate the final scores basing on the performance of the prediction process
			Args:
				bundled_data_and_scores: the data, iterator of (correct_rating, line, pred_rating)
				per_line_func: the function to execute per line, must be a callable fn(correct_rating, line, pred_rating)
				end_func: the function to run at the final, must be a callable fn(total, positivity, ratings, deviation)
			Returns:
				the result of the end_func. Rarely used
		"""
		# show the data by the bundled version
		assert end_func and callable(end_func), "Must have final return function"
		total_case = correct_pos_count = correct_rat_count = total_deviation = 0
		for correct_rating, line, pred_rating in bundled_data_and_scores:
			total_case += 1
			if(correct_rating == pred_rating):
				correct_rat_count += 1
				correct_pos_count += 1
			elif((correct_rating - 3) * (pred_rating - 3) > 0):
				correct_pos_count += 1
			total_deviation += abs(correct_rating - pred_rating)
			if(per_line_func):
				per_line_func(correct_rating, line, pred_rating)
		# show the heatmap as well
		return end_func(total_case, correct_pos_count, correct_rat_count, total_deviation)

	def tune(self, dataset, weight_correction=False, debug=False):
		"""Try to tune the session on this particular dataset
			Args:
				dataset: a dataset in form of list (sentence, rating)
				weight_correction: if not modified, the correct value will move by the rarity
				debug: if True, output certain informations
		"""
		# take all the placeholders
		session = self._session_dictionary["session"]
		input_pl, _, _ = self._session_dictionary["placeholders"]
		tune_inputs, tune_results, *_ = zip(*dataset)
		tune_results = [int(res) for res in tune_results]
		predictions_raw = self._session_dictionary["predictions_raw"]
		# batch and feed
		batched_tune_inputs = batch(list(tune_inputs), n=self._batch_size, shuffle_batch=False)
		tune_predictions = []
		for tune_batch in batched_tune_inputs:
			tune_preds = session.run(predictions_raw, feed_dict={input_pl:tune_batch})
			tune_predictions.extend(tune_preds)
		# reorders
		tune_set = list(zip(tune_results, tune_predictions))
		print("Size of tuning set: {:d}".format(len(tune_set)))
		# tune all the separating values between 1-2-3-4-5
		separators = []
		for sep in range(1, 5):
			# return False/True for 0.0 - 1.0, hence larger
			evaluation_function = lambda val: val > sep
			sep_value = tuneValuesSet(tune_set, evaluation_function, weight_correction=weight_correction, debug=debug)
			separators.append(sep_value)
		# create the tuned version of the rating
		predictions_tuned_rating = tf.convert_to_tensor(1, dtype=tf.int32, name="base")
		for sep_idx, sep_value in enumerate(separators):
			is_over_bracket = tf.greater(predictions_raw, tf.convert_to_tensor(sep_value, dtype=predictions_raw.dtype, name="brk{:d}-{:d}".format(sep_idx+1, sep_idx+2)))
			predictions_tuned_rating = tf.identity(predictions_tuned_rating + tf.cast(is_over_bracket, dtype=tf.int32), name="tuned_part_{:d}".format(sep_idx))
		# replace the untuned rating with the tuned one
		self._session_dictionary["predictions"] = predictions_tuned_rating
	
	def exportSession(self, export_sub_folder=0, override=False):
		"""Export the session for the tensorflow serving
			Args:
				export_sub_folder: the name of the subfolder where the exported model will be kept
		"""
		print("Using normal exportSession")
		session = self._session_dictionary["session"]
		input_pl, _, _ = self._session_dictionary["placeholders"]
		predictions = self._session_dictionary["predictions"]
		self._exportSession(session, {"input": input_pl}, {"output": predictions}, export_sub_folder=export_sub_folder, override=override)





class SentimentSelfAttention(SentimentRNNAttention):
	def __init__(self, save_path, export_path, batch_size=128, debug=False, shuffle_batch=True):
		self._save_path = save_path
		self._export_path = export_path
		self._batch_size = batch_size
		self._debug = debug
		self._shuffle_batch = shuffle_batch
		self._clear_word = "<cls>"
		self._loss_weight = None
		self._prev_stat = None

	def buildSession(self, table_words, table_vectors, table_default_idx, cell_size=512, depth=4, head=4, additional_words=None, gpu_allow_growth=True, optimizer="SGD"):
		# initialize the session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = gpu_allow_growth
		session = tf.Session(config=config)
		self._retrieveEpoch()
		# create the lookup word tablea
		# add an extra words among the additional words
		additional_words = [self._clear_word] + (additional_words if isinstance(additional_words, (list, tuple, set)) else [])
		word_table, embeddings_tensor = self._buildLookupTable(table_words, table_vectors, trainable_words=additional_words, default_word_idx=table_default_idx)
		# retrieve the ids of the additional word
		clear_id = word_table.lookup(tf.convert_to_tensor(self._clear_word))
		# create the inputs and lookup the ids by the table
		input_placeholder, input_tokenized_dense, input_length = self._buildInputSection()
		input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
		# add the clear id to the start of the indices list
		batch_size = tf.shape(input_indices)[0]
		batched_clear_ids = tf.fill([batch_size, 1], clear_id)
		input_indices = tf.concat([batched_clear_ids, input_indices], axis=1)
		input_length = input_length + 1
		# lookup the indices by the embeddings
		inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
		# encode their positions using the sinusoid scheme
		prev_depth = inputs = sinusoidEncoding(inputs)
		# construct the layers
		dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
		mask = tf.sequence_mask(input_length, name="mask")
		attentions = []
		for i in range(depth):
			layer_name = "layer_{:d}".format(i)
			next_depth, attention = createSelfAttentionLayer(prev_depth, input_length, cell_size, input_masks=mask, head=head, dropout_t=dropout_placeholder, name=layer_name)
			prev_depth = next_depth
			attentions.append(attention)
		# concat into [depth, batch_size, q_length, k_length], reduce to query of the clear id, and transpose into [batch_size, depth, k_length]
		attention = tf.concat(attentions, axis=0)[:, :, :1]
		attention = tf.transpose( tf.squeeze(attention, axis=[2]), perm=[1, 0, 2])
		# create the entropy using the last layer of depth
		result_placeholder, result_sigmoid = self._buildTrainingInputSection()
		predictions_raw, entropy = self._buildPredictionSection(prev_depth, result_sigmoid)
		# compute loss from predictions to true (result)
		predictions = tf.nn.sigmoid(predictions_raw, name="prediction_sigmoid")
		predictions_rating = tf.cast(1.5 + predictions * 4.0, dtype=tf.int32, name="predictions")
		# create train_op
		train_op, loss = self._createTrainingOp(entropy, optimizer, result_score=result_placeholder)
		# initialize
		self._session_dictionary = {
			"session": session, 
			"placeholders": (input_placeholder, result_placeholder, dropout_placeholder), 
			"training_ops":(loss, train_op), 
			"predictions_raw": predictions,
			"predictions": predictions_rating,
			"attentions": attention
		}
		return self._session_dictionary

	def _buildPredictionSection(self, last_layer, result_sigmoid):
		# take the first values, which correspond to the clear_ids
		output = last_layer[:, :1]
		output = tf.squeeze(output, axis=[1])
		# create the prediction using a simple projection, have bias
		pred_layer = tf.layers.Dense(1, name="pred")
		predictions_raw = tf.squeeze( pred_layer(output) , axis=[-1])
		# entropy
		entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_sigmoid, logits=predictions_raw)
		return predictions_raw, entropy


class SentimentRNNAttentionExtended(SentimentRNNAttention):
	def __init__(self, *args, **kwargs):
		"""Add _certainty_loss value"""
		self._certainty_loss = kwargs.pop("use_certainty_loss", False)
		super(SentimentRNNAttentionExtended, self).__init__(*args, **kwargs)

	def _buildTrainingInputSection(self):
		"""Override the _buildTrainingInputSection from parent
			Instead of one placeholder [batch_size] of int, we have [batch_size, 2] of float
			Receiving rating (1-5) and certainty (0-1)
		"""
		# RESULTS
		result_placeholder = tf.placeholder(tf.float32, shape=(None, 2), name="batch_result")
		result_score_placeholder = tf.squeeze(result_placeholder[:, :1], axis=[-1], name="batch_score")
		result_certainty_placeholder = tf.squeeze(result_placeholder[:, 1:2], axis=[-1], name="batch_certainty")
		return result_placeholder, result_score_placeholder, result_certainty_placeholder

	def _buildPredictionSection(self, rnn_output, rnn_length, rnn_size, result_tuple):
		"""Override the _buildPredictionSection from parent
			The result_tuple is actually a tuple of (result_score, result_certainty)
			Thus, the entropy is also the tuple of (entropy_positivity, entropy_intensity, entropy_certainty)
			Returns:
				a tuple of
					predictions: due to int/pos structure, a tuple of (predictions_raw, predictions_rating, predictions_sigmoids) of [batch_size] each, int(1-5) and float(0-1) 
					attentions: the attentions for int/pos/cer, [batch_size, 3, length]
					entropy: tuple of 3 above
		"""
		result_score, result_certainty = result_tuple
		result_intensity =  tf.abs(tf.cast(result_score, tf.float32) - 3.0) / 2.0
		result_positivity = tf.cast(result_score > 3, tf.float32)
		# the predictions aspects and their attentions
		predictions_intensity, intensity_attention = createEncoderAttention(rnn_size, rnn_output, rnn_length, scope="intensity")
		predictions_positivity, positivity_attention = createEncoderAttention(rnn_size, rnn_output, rnn_length, scope="positivity")
		predictions_certainty, certainty_attention = createEncoderAttention(rnn_size, rnn_output, rnn_length, scope="certainty")
		predictions_intensity_sig = tf.nn.sigmoid(predictions_intensity)
		predictions_positivity_sig = tf.nn.sigmoid(predictions_positivity)
		# sigmoid to 0-1, and turn into a rating
		predictions_intensity_rating = tf.cast(tf.round(2.0 * predictions_intensity_sig), tf.int32) # (scale 0-2)
		predictions_positivity_rating = tf.cast(tf.round(predictions_positivity_sig), tf.int32) * 2 - 1 # (scale -1/1)
		predictions_rating = tf.identity(3 + predictions_intensity_rating * predictions_positivity_rating, name="predictions")
		predictions_raw = tf.identity(0.5 + (tf.round(predictions_positivity_sig) - 0.5) * predictions_intensity_sig, name="raw")
		predictions_certainty_sig = tf.nn.sigmoid(predictions_certainty, name="pred_certainty")
		# attention joined together
		attention = tf.concat([tf.transpose(item, perm=[0, 2, 1]) for item in (intensity_attention, positivity_attention, certainty_attention)], axis=1)
		# the entropy
		entropy_intensity = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_intensity, logits=predictions_intensity)
		entropy_positivity_unmasked = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_positivity, logits=predictions_positivity)
		# for positivity, do not propel through those with neutral result (3)
		blank_mask = tf.cast(tf.fill(tf.shape(result_score), 0), dtype=entropy_positivity_unmasked.dtype)
		entropy_positivity = tf.where(result_score == 3, blank_mask, entropy_positivity_unmasked)
		entropy_certainty = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_certainty, logits=predictions_certainty)
		# return the values as in desc
		predictions_sig = predictions_intensity_sig, predictions_positivity_sig, predictions_certainty_sig
		return (predictions_raw, predictions_rating, predictions_sig), attention, (entropy_intensity, entropy_certainty, entropy_positivity)

	def _createTrainingOp(self, entropy, optimizer, certainty_value=None, result_score=None):
		"""Override the _createTrainingOp of parent
			entropy is a tuple of three, trying to minimize it
			we can either sum it or penalize those with low certainty and wrong. this mode will be self._certainty_loss
			Extra Args:
				certainty_value: the value of predicted certainty by the model. [batch_size]
				result_score: the value of the scores fed during training. [batch_size]
		"""
		entropy_positivity, entropy_intensity, entropy_certainty = entropy
		entropy = entropy_positivity + entropy_intensity
		if(self._certainty_loss):
			print("Use certainty loss")
			assert certainty_value is not None, "_certainty_loss mode must have certainty_value"
			# balance point at around 0.69, increase entropy exponentially
			entropy = tf.exp(entropy - 2 * 0.5) * (1.0 - certainty_value * 0.5)
		else:
			entropy = entropy + entropy_certainty
		return super(SentimentRNNAttentionExtended, self)._createTrainingOp(entropy, optimizer, result_score=result_score)

	def buildSession(self, table_words, table_vectors, table_default_idx, cell_size=512, additional_words=None, gpu_allow_growth=True, optimizer="SGD"):
		"""An extended version from the buildSessionBidirectionalAttention, but instead of 1 there is now 3 prediction values: positivity, intensity and certainty, basing on the data of the comment.
			Args:
				table_words: list, words loaded from the embedding reader
				table_vectors: numpy array 2d, vectors correlating with the words
				table_default_idx: int, the index of the <unk> token
				cell_size: the size of the encoder cell
				additional_words: a list of additional words to be added into table_words if exist. Their vectors will be randomized and trained unlike the table_vectors
				gpu_allow_growth: if set True, the tensorflow will only expand to the size it needs
			Returns:
				dict of
					session: finished tf.Session with the graph initialized
					placeholders: the placeholders used to either train or evaluate. inputs(str), outputs( tuple of float 0.0-1.0), dropout(float 0.0-1.0)
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
		self._retrieveEpoch()
		# anchor the embedding as constants
		word_table, embeddings_tensor = self._buildLookupTable(table_words, table_vectors, trainable_words=additional_words, default_word_idx=table_default_idx)
		# create the input placeholder, tokenize and feed through table
		input_placeholder, input_tokenized_dense, input_length = self._buildInputSection()
		# lookup the ids by the table
		input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
		# create embedding and lookup the indices
		inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
		# MISC
		dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
		# run bidirectional RNN 
		outputs, last_state = builder.createEncoderClean(inputs, cell_size=cell_size, created_dropout=dropout_placeholder)
		# the correct values
		result_placeholder, result_score, result_certainty = self._buildTrainingInputSection()
		result_tuple = (result_score, result_certainty)
		# build the prediction section
		(predictions_raw, predictions_rating, predictions_sigmoids), attention, entropy = self._buildPredictionSection(outputs, input_length, cell_size*2, result_tuple)
		# create train_op
		train_op, loss = self._createTrainingOp(entropy, optimizer, certainty_value=predictions_sigmoids[-1], result_score=result_score)
		# initialize
		self._session_dictionary = {
			"session": session, 
			"placeholders": (input_placeholder, result_placeholder, dropout_placeholder), 
			"training_ops":(loss, train_op), 
			"predictions": predictions_rating,
			"predictions_raw": predictions_raw,
			"predictions_sigmoids": predictions_sigmoids,
			"attentions": attention
		}
		self._attention_names = ["intensity", "positivity", "certainty"]
		return self._session_dictionary

	def trainSession(self, training_dataset, eval_dataset, epoch, dropout=1.0):
		session_dictionary = self._session_dictionary
		session = session_dictionary["session"]
		input_pl, result_pl, dropout_pl = session_dictionary["placeholders"]
		loss_t, train_op = trainers = session_dictionary["training_ops"] 
		training_dropout = dropout
		timer = time.time()
		# bundle the score and the abs reliability rating together
		training_dataset = [ (line, (score, rel_rating)) for line, score, rel_rating in training_dataset ]
		for i in range(self._epoch, epoch):
			# start training
			batch_iters = batch(training_dataset, n=self._batch_size, shuffle_batch=self._shuffle_batch)
			for batch_idx, iter_batch in enumerate(batch_iters):
				inputs, correct_outputs = zip(*iter_batch)
				correct_outputs = np.asarray(correct_outputs, dtype=np.float32)
				assert np.shape(correct_outputs)[1] == 2, "In extended output mode, the input must be a tuple of score-certainty"
				inputs = list(inputs)
				loss, _ = session.run(trainers, feed_dict={input_pl:inputs, result_pl:correct_outputs, dropout_pl:training_dropout})
				if(self._debug):
					print("Iter {:d}, batch {:d}, loss {:.4f}, time passed {:.2f}".format(i, batch_idx, loss, time.time() - timer))
			print("Epoch iter {:d} finished, time passed {:.2f}".format(i, time.time() - timer))
			# start eval if eval dataset available
			if(not eval_dataset):
				continue
			self._epoch += 1
			self._current_stat = self.evalSession(eval_dataset, detailed=False)
			self._prev_stat = self.saveSession(self._default_compare_function, model_idx=(i+2))
		print("All training phase completed, time passed {:.2f}".format(time.time() - timer))

	def exportSession(self, export_sub_folder=0, override=False):
		print("Using export Attention Extended")
		session = self._session_dictionary["session"]
		input_pl, _, _ = self._session_dictionary["placeholders"]
		predictions = self._session_dictionary["predictions"]
		predictions_raw = self._session_dictionary["predictions_raw"]
		self._exportSession(session, {"input": input_pl}, {"output": predictions_raw, "output_rating": predictions}, export_sub_folder=export_sub_folder, override=override)

	def tune(self, dataset, weight_correction=False, debug=False):
		"""
			Override from the parent (SentimentRNNAttention) as well
			require the sigmoid'd 3 predictions
		"""
		# take all the placeholders
		session = self._session_dictionary["session"]
		input_pl, _, _ = self._session_dictionary["placeholders"]
		tune_inputs, tune_results, *_ = zip(*dataset)
		tune_results = [int(res) for res in tune_results]
		predictions_sigmoids = self._session_dictionary["predictions_sigmoids"]
		# batch and feed
		batched_tune_inputs = batch(list(tune_inputs), n=self._batch_size, shuffle_batch=False)
		tune_predictions = []
		for tune_batch in batched_tune_inputs:
			tune_preds = session.run(predictions_sigmoids, feed_dict={input_pl:tune_batch})
			tune_predictions.extend(zip(*tune_preds))
		# tune the 2-4/1-5 difference through intensity, 1-2/4-5 difference through positivity and 3/1-2-4-5 through certainty
		# split into bunches
		intensity_preds, positivity_preds, certainty_preds = zip(*tune_predictions)
		# intensity
		intensity_tune_set = list(zip(tune_results, intensity_preds))
		intensity_exist_fn = lambda val: val != 3
		intensity_is_high_fn = lambda val: val == 1 or val == 5 
		intensity_neutral_sep = tuneValuesSet(intensity_tune_set, intensity_exist_fn, weight_correction=weight_correction, debug=debug)
		intensity_fine_sep = tuneValuesSet(intensity_tune_set, intensity_is_high_fn, weight_correction=weight_correction, debug=debug)
		# positivity set must remove neutral
		positivity_tune_set = [it for it in zip(tune_results, positivity_preds) if it[0] != 3]
		positivity_fn = lambda val: val > 3
		positivity_sep = tuneValuesSet(positivity_tune_set, positivity_fn, weight_correction=weight_correction, debug=debug)
		# create the tuned rating
		predictions_intensity, predictions_positivity, predictions_certainty = predictions_sigmoids
		predictions_base = tf.convert_to_tensor(3, dtype=tf.int32, name="base")
		neutral_modifier = tf.cast(tf.greater(predictions_intensity, tf.convert_to_tensor(intensity_neutral_sep, dtype=predictions_intensity.dtype, name="neutral_sep")), dtype=tf.int32, name="neutral_mod")
		intensity_modifier = neutral_modifier + tf.cast(tf.greater(predictions_intensity, tf.convert_to_tensor(intensity_fine_sep, dtype=predictions_intensity.dtype, name="fine_sep")), dtype=tf.int32, name="fine_mod") # 0/1/2 value
		positivity_modifier = tf.identity( tf.cast(tf.greater(predictions_positivity, tf.convert_to_tensor(positivity_sep, dtype=predictions_positivity.dtype, name="pos_sep")), dtype=tf.int32) * 2 - 1 , name="pos_mod") # -1/1 value
		predictions_tuned_rating = predictions_base + intensity_modifier * positivity_modifier
		# replace the untuned rating with the tuned one
		self._session_dictionary["predictions"] = predictions_tuned_rating





class SentimentRNNAttentionExtendedV2(SentimentRNNAttentionExtended):
	def _buildPredictionSection(self, rnn_output, rnn_length, rnn_size, result_tuple):
		result_score, _ = result_tuple
		result_intensity = tf.maximum( tf.abs(tf.cast(result_score, tf.float32) - 3.0) - 1.0, 0.0 )
		result_positivity = tf.cast(result_score > 3, tf.float32)
		result_certainty = tf.cast(tf.not_equal(result_score, 3), tf.float32)
		# the predictions aspects and their attentions
		predictions_intensity, intensity_attention = createEncoderAttention(rnn_size, rnn_output, rnn_length, scope="intensity")
		predictions_positivity, positivity_attention = createEncoderAttention(rnn_size, rnn_output, rnn_length, scope="positivity")
		predictions_certainty, certainty_attention = createEncoderAttention(rnn_size, rnn_output, rnn_length, scope="certainty")
		predictions_intensity_sig = tf.nn.sigmoid(predictions_intensity)
		predictions_positivity_sig = tf.nn.sigmoid(predictions_positivity)
		predictions_certainty_sig = tf.nn.sigmoid(predictions_certainty, name="pred_certainty")
		# sigmoid to 0-1, and turn into a rating
		predictions_intensity_rating = tf.cast(tf.round(predictions_intensity_sig), tf.int32) + 1 # (scale 1-2)
		predictions_positivity_rating = tf.cast(tf.round(predictions_positivity_sig), tf.int32) * 2 - 1 # (scale -1/1)
		predictions_certainty_rating = tf.cast(tf.round(predictions_certainty_sig), tf.int32) # (scale 0-1)
		predictions_rating = tf.identity(3 + predictions_certainty_rating * predictions_intensity_rating * predictions_positivity_rating, name="predictions")
		predictions_raw = tf.identity(0.5 + (tf.round(predictions_positivity_sig) - 0.5) * predictions_intensity_sig, name="raw")
		# attention joined together
		attention = tf.concat([tf.transpose(item, perm=[0, 2, 1]) for item in (intensity_attention, positivity_attention, certainty_attention)], axis=1)
		# the entropy
		entropy_certainty = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_certainty, logits=predictions_certainty)
		entropy_intensity_unmasked = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_intensity, logits=predictions_intensity)
		entropy_positivity_unmasked = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_positivity, logits=predictions_positivity)
		# for positivity and intensity, do not propel through those with neutral result (3)
		blank_mask = tf.cast(tf.fill(tf.shape(result_score), 0), dtype=entropy_positivity_unmasked.dtype)
		entropy_positivity = tf.where(result_score == 3, blank_mask, entropy_positivity_unmasked)
		entropy_intensity = tf.where(result_score == 3, blank_mask, entropy_intensity_unmasked)
		# return the values as in desc
		predictions_sig = predictions_intensity_sig, predictions_positivity_sig, predictions_certainty_sig
		return (predictions_raw, predictions_rating, predictions_sig), attention, (entropy_intensity, entropy_certainty, entropy_positivity)
	
	def tune(self, dataset, weight_correction=False, debug=False):
		# take all the placeholders
		session = self._session_dictionary["session"]
		input_pl, _, _ = self._session_dictionary["placeholders"]
		tune_inputs, tune_results, *_ = zip(*dataset)
		tune_results = [int(res) for res in tune_results]
		predictions_sigmoids = self._session_dictionary["predictions_sigmoids"]
		# batch and feed
		batched_tune_inputs = batch(list(tune_inputs), n=self._batch_size, shuffle_batch=False)
		tune_predictions = []
		for tune_batch in batched_tune_inputs:
			tune_preds = session.run(predictions_sigmoids, feed_dict={input_pl:tune_batch})
			tune_predictions.extend(zip(*tune_preds))
		# tune the 3/2-4/1-5 difference through intensity and 1-2/4-5 difference through positivity
		# split into bunches
		intensity_preds, positivity_preds, certainty_preds = zip(*tune_predictions)
		# intensity must remove neutral 
		intensity_tune_set = [it for it in zip(tune_results, intensity_preds) if it[0] != 3]
		intensity_is_high_fn = lambda val: val == 1 or val == 5 
		intensity_sep = tuneValuesSet(intensity_tune_set, intensity_is_high_fn, weight_correction=weight_correction, debug=debug)
		# positivity set must remove neutral
		positivity_tune_set = [it for it in zip(tune_results, positivity_preds) if it[0] != 3]
		positivity_fn = lambda val: val > 3
		positivity_sep = tuneValuesSet(positivity_tune_set, positivity_fn, weight_correction=weight_correction, debug=debug)
		# certainty
		certainty_tune_set = list( zip(tune_results, certainty_preds) )
		certainty_fn = lambda val: val != 3
		certainty_sep = tuneValuesSet(certainty_tune_set, certainty_fn, weight_correction=weight_correction, debug=debug)
		# create the tuned rating
		predictions_intensity, predictions_positivity, predictions_certainty = predictions_sigmoids
		predictions_base = tf.convert_to_tensor(3, dtype=tf.int32, name="base")
		certainty_modifier = tf.cast(tf.greater(predictions_certainty, tf.convert_to_tensor(certainty_sep, dtype=predictions_certainty.dtype, name="cer_sep")), dtype=tf.int32, name="cer_mod") # 0/1 value
		intensity_modifier = 1 + tf.cast(tf.greater(predictions_intensity, tf.convert_to_tensor(intensity_sep, dtype=predictions_intensity.dtype, name="int_sep")), dtype=tf.int32, name="int_mod") # 1/2 value
		positivity_modifier = tf.identity( tf.cast(tf.greater(predictions_positivity, tf.convert_to_tensor(positivity_sep, dtype=predictions_positivity.dtype, name="pos_sep")), dtype=tf.int32) * 2 - 1 , name="pos_mod") # -1/1 value
		predictions_tuned_rating = predictions_base + certainty_modifier * intensity_modifier * positivity_modifier
		# replace the untuned rating with the tuned one
		self._session_dictionary["predictions"] = predictions_tuned_rating
		print("Tuning completed.")





class SentimentRNNAttentionMultimodal(SentimentRNNAttentionExtended):
	def __init__(self, *args, **kwargs):
		super(SentimentRNNAttentionMultimodal, self).__init__(*args, **kwargs)
		# prepare for eval
		self._first_eval = True

	def buildSession(self, table_words, table_vectors, table_default_idx, cell_size=512, additional_words=None, gpu_allow_growth=True, optimizer="SGD"):
		"""A multimodal version from the buildSessionBidirectionalAttention, softmaxing values on 5 class instead of relying on single sigmoid
			Args:
				table_words: list, words loaded from the embedding reader
				table_vectors: numpy array 2d, vectors correlating with the words
				table_default_idx: int, the index of the <unk> token
				cell_size: the size of the encoder cell
				additional_words: a list of additional words to be added into table_words if exist. Their vectors will be randomized and trained unlike the table_vectors
				gpu_allow_growth: if set True, the tensorflow will only expand to the size it needs
			Returns:
				dict of
					session: finished tf.Session with the graph initialized
					placeholders: the placeholders used to either train or evaluate. inputs(str), outputs( tuple of float 0.0-1.0), dropout(float 0.0-1.0)
					training ops: the loss and train operation for training mode. tensor (float) and op
					predictions: a float tensor of [batch_size, num_class denoting the possibility of each]
		"""
		raise NotImplementedError("Unfinished refactoring")
		# initialize the session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = gpu_allow_growth
		session = tf.Session(config=config)
		self._retrieveEpoch()
		# anchor the embedding as constants
		word_table, embeddings_tensor = self._buildLookupTable(table_words, table_vectors, trainable_words=additional_words, default_word_idx=table_default_idx)
		input_placeholder, input_tokenized_dense, input_length = self._buildInputSection()
		# lookup the ids by the table
		input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
		# create embedding and lookup the indices
		inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
		# RESULTS
		result_placeholder = tf.placeholder(tf.float32, shape=(None, 2), name="batch_result")
		result_score_placeholder = tf.squeeze(result_placeholder[:, :1], axis=[-1], name="batch_score")
		result_certainty_placeholder = tf.squeeze(result_placeholder[:, 1:2], axis=[-1], name="batch_certainty")
		# hack in by a gather, create the probs distribution similar to a smoothing label basing on the primary label and the reliability
		off_value_probs = tf.expand_dims((1.0 - result_certainty_placeholder) / (5.0 - 1.0), axis=-1)
		on_value_probs = tf.expand_dims(result_certainty_placeholder, axis=-1)
		result_probs_values = tf.concat([off_value_probs, on_value_probs], axis=-1)
#		result_score_placeholder = tf.Print(result_score_placeholder, [tf.shape(result_probs_values)])
		result_score_index = tf.cast(result_score_placeholder, tf.int32) - 1
		with tf.control_dependencies([tf.assert_greater_equal(result_score_index, 0), tf.assert_less(result_score_index, 5)]):
			result_probs_idx = tf.one_hot(result_score_index, 5, on_value=1, off_value=0) + tf.expand_dims(tf.range(tf.shape(result_score_placeholder)[0]), axis=-1) * 2
		correct_shape = tf.shape(result_probs_idx)
		result_flatten_probs = tf.gather(tf.reshape(result_probs_values, [-1]), tf.reshape(result_probs_idx, [-1]))
		result_true_probs = tf.reshape(result_flatten_probs, correct_shape, name="batch_true_probs")
		# MISC
		dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
		# run bidirectional RNN 
		outputs, last_state = builder.createEncoderClean(inputs, cell_size=cell_size, created_dropout=dropout_placeholder)
		# the predictions in raw (un-softmaxed logits)
		predictions_raw = createEncoderMultimodal(5, cell_size * 2, outputs, input_length, scope="multimodal_preds")
		predictions = tf.argmax(predictions_raw, axis=-1, name="predictions") + 1
		with tf.control_dependencies([tf.assert_equal(tf.shape(result_true_probs), tf.shape(predictions_raw)), tf.assert_near(tf.reduce_sum(result_true_probs, axis=-1), 1.0)]):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=result_true_probs, logits=predictions_raw)
	#	loss = tf.reduce_sum(cross_entropy, axis=-1)
		# create train_op
		train_op, loss = self._createTrainingOp(cross_entropy, optimizer)
		# initialize
		self._session_dictionary = {
			"session": session, 
			"placeholders": (input_placeholder, result_placeholder, dropout_placeholder), 
			"training_ops":(loss, train_op), 
			"predictions": predictions
		}
		return self._session_dictionary

	def evalSession(self, dataset, detailed=False):
		# Use the eval of original version
		# TODO maybe do it in the form of Extended?
		# rip the reliability rating out
		line, score, rel = zip(*dataset)
		dataset = zip(line, score)
		# port the prediction from 0-4 to 0.0-1.0 range
		if(self._first_eval):
			print("Changing prediction value for eval")
			self._session_dictionary["predictions_backup"] = self._session_dictionary["predictions"]
			self._session_dictionary["predictions"] = (tf.cast(self._session_dictionary["predictions"], dtype=tf.float32) - 1.45) / 4.0
			self._first_eval = False
		return SentimentRNNAttention.evalSession(self, dataset, detailed=detailed)




class SentimentRNNBalanceModel(SentimentRNNAttention):
	def _buildPredictionSection(self, outputs, input_length, cell_size, result_sigmoid):
		"""Override the _buildPredictionSection of Parent
			Use combined sum instead of attention
		"""
		outputs_score_layer = tf.layers.Dense(1, use_bias=True, name="scoring_layer")
		scored_outputs = tf.squeeze(outputs_score_layer(outputs), axis=[-1])
		mask = tf.sequence_mask(input_length, maxlen=tf.shape(scored_outputs)[1], dtype=scored_outputs.dtype)
		masked_outputs = scored_outputs * mask
		predictions_raw = tf.reduce_sum(masked_outputs, axis=-1, name="predictions_raw")
		entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_sigmoid, logits=predictions_raw)
		# we can use masked outputs as attention, since the heatmap scale regardless of range
		attention = tf.expand_dims(masked_outputs, 1)
		return predictions_raw, attention, entropy

def createEncoderAttention(attention_cell_size, encoder_outputs, encoder_output_lengths, scope="default"):
	"""Apply attention on top of the encoder outputs
		Args:
			attention_cell_size: int, must match encoder_outputs last dim (num_units)
			encoder_outputs: size [batch, length, num_units]
			encoder_length: size [batch]
		Returns:
			The unscaled prediction logits [batch], the attention tensor [batch, length]
	"""
	with tf.variable_scope(scope):
		attention_base = tf.get_variable("attention_base", shape=[1, attention_cell_size, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
		batch_size = tf.shape(encoder_outputs)[0]
		attention_batch = tf.tile(attention_base, [batch_size, 1, 1])
		# compare directly, and mask
		# matrix multiply: [batch, length, num_units] * [batch, num_units, 1] = [batch, length, 1]
		# the encoder outputs should be normalized?
		outputs = tf.nn.l2_normalize(encoder_outputs, axis=-1)
#		outputs = encoder_outputs
		attention_unmasked = tf.matmul(outputs, attention_batch)
		# masking log values, so select -inf if false
		mask_choice = tf.sequence_mask(encoder_output_lengths, maxlen=tf.shape(encoder_outputs)[1])
		mask_values = tf.fill(tf.shape(attention_unmasked), tf.float32.min)
		mask_choice = tf.expand_dims(mask_choice, axis=-1)
		# apply the mask and get the attention
		attention_masked = tf.where(mask_choice, attention_unmasked, mask_values)
		attention = tf.nn.softmax(attention_masked, axis=1, name="attention")
		# compute context
		context_raw = tf.multiply(attention, encoder_outputs)
		context = tf.reduce_sum(context_raw, axis=1, name="context")
		# feed this context through an activation layer to get result
		predictions_raw = tf.layers.dense(context, 1)
		predictions_raw = tf.squeeze(predictions_raw, axis=[-1], name="prediction_raw")
	return predictions_raw, attention

def createEncoderMultimodal(labels_num, attention_cell_size, encoder_outputs, encoder_output_lengths, scope="default"):
	"""Apply attention on top of the encoder outputs and create a softmax over it
		Args:
			labels_num: the number of labels to output the logits of
			attention_cell_size: int, must match encoder_outputs last dim (num_units)
			encoder_outputs: size [batch, length, num_units]
			encoder_length: size [batch]
		Returns:
			The unscaled prediction logits [batch]
	"""
	with tf.variable_scope(scope):
		attention_base = tf.get_variable("attention_base", shape=[1, attention_cell_size, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
		batch_size = tf.shape(encoder_outputs)[0]
		attention_batch = tf.tile(attention_base, [batch_size, 1, 1])
		# compare directly, and mask
		# matrix multiply: [batch, length, num_units] * [batch, num_units, 1] = [batch, length, 1]
		attention_unmasked = tf.matmul(encoder_outputs, attention_batch)
		# masking log values, so select -inf if false
		mask_choice = tf.sequence_mask(encoder_output_lengths, maxlen=tf.shape(encoder_outputs)[1])
		mask_values = tf.fill(tf.shape(attention_unmasked), tf.float32.min)
		mask_choice = tf.expand_dims(mask_choice, axis=-1)
		# apply the mask and get the attention
		attention_masked = tf.where(mask_choice, attention_unmasked, mask_values)
		attention = tf.nn.softmax(attention_masked, name="attention")
		# compute context
		context_raw = tf.multiply(attention, encoder_outputs)
		context = tf.reduce_sum(context_raw, axis=1, name="context")
		# feed this context through an activation layer to get result
		predictions_raw = tf.layers.dense(context, labels_num)
	return predictions_raw

def batch(iterable, n=1, shuffle_batch=False):
	l = len(iterable)
	true_range = range(0, l, n)
	if(shuffle_batch):
		true_range = list(true_range)
		shuffle(true_range)
	for ndx in true_range:
		yield iterable[ndx:min(ndx + n, l)]

def tuneValuesSet(zipped_set, evaluation_fn, weight_correction=False, debug=False):
	"""For a predetermined set in range 0.0-1.0, find a value within that range splitting the set into its best value
		Args:
			zipped_set: the list of (eval_value, sigmoid_value) to be tuned
			evaluation_fn: function taking eval_value and return False/True for 0.0-1.0 . Callable
			weight_correction: if true, weight eval_value inversely propotionate to its occurence
			debug: if true, output relevant info
		Return:
			split_value of float
	"""
	current_correct, best_sep, best_sep_count = 0.0, -1, 0.0
	val_range, pred_range = zip(*sorted(zipped_set, key=lambda it: it[-1]))
#	if(debug):
#		print(val_range)
#		print(pred_range)
	if(weight_correction):
		get_correct_fn = _createWeightCorrectionFn(val_range, debug=debug)
	else:
		get_correct_fn = lambda it: 1.0
	for idx, val in enumerate(val_range):
		if(not evaluation_fn(val)):
			# moving right absorbed good value (False)
			current_correct += get_correct_fn(val)
		else:
			# bad value (True)
			current_correct -= get_correct_fn(val)
		if(current_correct > best_sep_count):
			# override
			best_sep_count = current_correct
			best_sep = idx
	limit_left = pred_range[best_sep] if best_sep >= 0 else 0.0
	limit_right = pred_range[best_sep + 1] if(best_sep + 1 < len(pred_range)) else 1.0
	if(debug):
		print("Separator found at index {:d}, count {:.3f}, between {:.4f}-{:.4f}".format(best_sep, best_sep_count, limit_left, limit_right ))
	return (limit_left + limit_right) / 2

def _createWeightCorrectionFn(val_range, debug=False):
	# correct the weights
	rat_dict = {}
	rat_size = float(len(val_range))
	for rat in val_range:
		rat_dict[rat] = rat_dict.get(rat, 0.0) + 1.0
	rat_dict = {k:rat_size/v for k, v in rat_dict.items()}
	if(debug):
		print("Weight correction rat_dict: {}".format(rat_dict))
	return lambda val: rat_dict[val]

def _createSimpleTokenizationOperations(strings, op_name="tokenized"):
	from scripts.tokenize_word import spacer, space_reductor
	spacer, space_reductor = spacer.pattern(), space_reductor.pattern()
	# first, add spaces to all
	strings = tf.regex_replace(strings, spacer, " \\g<0> ", name="add_spacing")
	strings = tf.regex_replace(strings, space_reductor, " ", name=op_name)
	return strings

def _splitHead(tensor, head, embedding_size):
	"""Reshape and transpose [batch_size, length, emb] into [batch_size, head, length, head_size]"""
	batch_size = tf.shape(tensor)[0]
	length = tf.shape(tensor)[1]
	head_size = embedding_size // head
	tensor = tf.reshape(tensor, [batch_size, length, head, head_size])
	return tf.transpose(tensor, [0, 2, 1, 3])

def _rejoinHead(tensor, head, embedding_size):
	tensor = tf.transpose(tensor, [0, 2, 1, 3])
	batch_size = tf.shape(tensor)[0]
	length = tf.shape(tensor)[1]
	return tf.reshape(tensor, [batch_size, length, embedding_size])

def createSelfAttentionLayer(inputs, input_lengths, embedding_size, input_masks=None, head=4, dropout_t=None, name="default"):
	"""Create a layer of self attention 
		Args:
			inputs: the input, [batch_size, length, embedding_size]
			input_lengths: the length of each sentences in the input, [batch_size]
			embedding_size: the size of the embedding. Must be int, for the Dense layers initialization
			input_mask: if specified, use this instead of tf.sequence_mask(input_lengths). [batch_size, length]
			head: the head to split, int
		Returns:
			the output [batch_size, length, embedding_size] and attention [batch_size, length, length]
	"""
	query_layer = tf.layers.Dense(embedding_size, use_bias=False, name=name+"_q")
	key_layer = tf.layers.Dense(embedding_size, use_bias=False, name=name+"_k")
	
	query = _splitHead( query_layer(inputs), head, embedding_size )
	key = _splitHead( key_layer(inputs), head, embedding_size )
	value = _splitHead( inputs, head, embedding_size )
	# scale query, check https://github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py for why
	# may or may not have a positive effect on the situation
#	head_size = embedding_size // head
	logits = tf.matmul(query, key, transpose_b=True)
	attention = tf.nn.softmax(logits, name=name+"_attention")
	context = _rejoinHead( tf.matmul(attention, value), head, embedding_size)
	# rejoin the context alongside the inputs
	output_layer = tf.layers.Dense(embedding_size, use_bias=False, name=name+"_o")
	concat_values = tf.concat([inputs, context], axis=2)
	if(dropout_t is not None):
		# add dropout over the concaternated values
		concat_values = tf.layers.dropout(concat_values, rate=dropout_t, name=name+"_do")
	outputs = output_layer(concat_values)
	return outputs, attention

def sinusoidEncoding(inputs, scaling_range=(1.0, 1.0e4)):
	"""Run the inputs through a cos-sin positional encoding. Used for self-attention
		Args:
			inputs: tensor of [batch_size, length, embedding_size]
			scaling_range: the range for the positional encoding used. Tuple of (float, float)
		Returns:
			the sinusoid version of the inputs
	"""
	# unload values
	length = tf.shape(inputs)[1]
	hidden_size = tf.shape(inputs)[2]
	min_scaling, max_scaling = scaling_range
	# necessary additional vals
	num_increases = hidden_size // 2
	increment = tf.log(max_scaling / min_scaling) / tf.to_float(num_increases - 1)
	# scaling made from this increment
	# inverse is e^(min_scale + hidden_size_pos*-inc), so the bigger the index, the higher this inverse
	# must be used to differentiate
	inverse_base = tf.cast(tf.range(num_increases), dtype=inputs.dtype, name="inv")
	inverse_scaling = min_scaling * tf.exp(inverse_base * -increment)
	# position is the position in length
	position = tf.cast(tf.range(length), dtype=inputs.dtype, name="position")
	# merge them together, result in [length, hidden_size/2] mask
	base_time = tf.expand_dims(position, 1) * tf.expand_dims(inverse_scaling, 0)
	# sinusoid it away to get the fitting mask with size [length, hidden_size]
	positional_sinusoid = tf.concat([tf.sin(base_time), tf.cos(base_time)], axis=-1, name="sinusoid")
	# add it directly into the inputs
	return inputs + tf.expand_dims(positional_sinusoid, 0)
