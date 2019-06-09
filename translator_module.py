import tensorflow as tf
import io, re
import subprocess
import translator_hooks

UNK_ID, UNK_TOKEN = 0, "<unk>"
SOS_ID, SOS_TOKEN = 1, "<s>"
EOS_ID, EOS_TOKEN = 2, "</s>"
def _check_vocab_file(file_path):
	tf.logging.debug("Checking if {} is a valid file path (first three lines is <unk>, <s>, </s>)".format(file_path))
	with io.open(file_path, "r", encoding="utf-8") as vocab_file:
		lines = vocab_file.readlines()
		assert lines[UNK_ID].strip() == UNK_TOKEN, lines[UNK_ID]
		assert lines[SOS_ID].strip() == SOS_TOKEN, lines[SOS_ID]
		assert lines[EOS_ID].strip() == EOS_TOKEN, lines[EOS_ID]
		return len(lines)

def check_blank_lines(file_path):
	tf.logging.debug("Check if file {:s} containing blank".format(file_path))
	with io.open(file_path, "r", encoding="utf-8") as check_file:
		for idx, line in enumerate(check_file.readlines()):
			assert line.strip() != "", "Evaluation files must NOT contain blank line; but line {:d} is blank".format(idx)

def _simple_processing_dataset(dataset):
	# string_split do not work on singular string, so make a 1-d string out of it. squeeze later
	dataset = dataset.map( lambda features, labels: (tf.string_split([features]), tf.string_split([labels])) )
	add_length = lambda item: (item, tf.size(item))
	dataset = dataset.map( lambda features, labels: (add_length(features), add_length(labels)) )
	# force to dense and squeeze the first (empty) dimension to allow batching in later stages. use spare_to_dense for compatibility with 1.7
	# the <pad> token is never used. TODO check if there exist padded values
	force_to_dense = lambda item: (tf.squeeze(tf.sparse_tensor_to_dense(item[0], default_value="<pad>"), axis=[0]), item[1])
	dataset = dataset.map( lambda features, labels: (force_to_dense(features), force_to_dense(labels)) )
	return dataset

def _build_attention(rnn_cell, mode, batch_size, beam_size, scope="cell", wrapper_config=None):
	"""Build attention wrapping depending on the mode.
	Args:
		rnn_cell: the cell to be wrapped by attention
		mode: the mode of the model_fn
		batch_size: the size of the data batch
		beam_size: the width of the beam used
		scope: the scope to build the cell into
		wrapper_config: tuple of num_units, memory, memory_sequence_length, rnn_initial_state
	Returns:
		tuple of single_cell, single_start_state, beamed_cell, beamed_start_state
	"""
	num_units, memory, memory_sequence_length, rnn_state = wrapper_config
	dtype = memory.dtype
	# alright, this is outright retarded.
	attention_layer = tf.layers.Dense(num_units, dtype=dtype, name="attention_layer")
	if(mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
		with tf.variable_scope(scope, reuse=False) as single_scope:
			tf.logging.debug("Building attention cell for logits (TRAIN/EVAL) in scope {}".format(tf.contrib.framework.get_name_scope()))
			# as first, initialize all these from scratch
			single_attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, memory=memory, memory_sequence_length=memory_sequence_length)
			single_wrapper = tf.contrib.seq2seq.AttentionWrapper(rnn_cell, single_attention_mechanism, attention_layer_size=num_units)
			single_start_state = single_wrapper.zero_state(batch_size, dtype).clone(cell_state=rnn_state)
	else:
		single_wrapper, single_start_state = None, None
		single_scope = None

	if(mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL):
		with tf.variable_scope(single_scope, default_name=scope, reuse=(mode == tf.estimator.ModeKeys.EVAL)):
			tf.logging.debug("Building attention cell for infer (PREDICT/EVAL) in scope {}".format(tf.contrib.framework.get_name_scope()))
			# only reuse if the mode is eval (single version created)
			# first, tile everything
			tiled_memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_size)
			tiled_memory_sequence_length = tf.contrib.seq2seq.tile_batch(memory_sequence_length, multiplier=beam_size)
			tiled_state = tf.contrib.seq2seq.tile_batch(rnn_state, multiplier=beam_size)
			# construct the wrapper and states as normal
			beamed_attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, memory=tiled_memory, memory_sequence_length=tiled_memory_sequence_length)
			beamed_wrapper = tf.contrib.seq2seq.AttentionWrapper(rnn_cell, beamed_attention_mechanism, attention_layer_size=num_units, alignment_history=True)
			beamed_start_state = beamed_wrapper.zero_state(batch_size * beam_size, dtype).clone(cell_state=tiled_state)
	else:
		beamed_wrapper, beamed_start_state = None, None
	# return the cells
	return single_wrapper, single_start_state, beamed_wrapper, beamed_start_state

def _replace_unk_tokens(tgt_tokens, tgt_ids, alignments, src_tokens):
	"""Replace the unk tokens with the highest aligned token in source
	Args:
		tgt_tokens: the reordered and mapped target tokens, still with UNK_TOKEN. [batch_size, beam_width, tgt_len]
		tgt_ids: the reordered ids, [batch_size, beam_width, tgt_len]
		alignments: the alignments recovered from the beam process, [batch_size, beam_width, tgt_len, src_len]
		src_tokens: the tokens from the source [batch_size, src_len]
	Returns:
		the reordered tgt_tokens with the replaced pieces
	"""
	replacer = tf.equal(tgt_ids, UNK_ID)
	aligned_src_indices = tf.argmax(alignments, axis=-1, name="aligned_src_indices_raw")
	# increase the indices to allow gathering
	batch_size = tf.shape(tgt_ids)[0]
	src_len = tf.shape(src_tokens)[-1]
	aligned_increase = tf.range(batch_size) * src_len
	aligned_src_indices = tf.cast(aligned_src_indices, aligned_increase.dtype) + tf.reshape(aligned_increase, [batch_size, 1, 1])
	# gather the correlating tokens from src
	flattened_src_tokens = tf.reshape(src_tokens, [-1])
	aligned_src_tokens = tf.gather(flattened_src_tokens, aligned_src_indices, name="aligned_src_tokens")
	# replace the <unk> token with the aligned src
	replaced_tgt_tokens = tf.where(replacer, x=aligned_src_tokens, y=tgt_tokens)
	return replaced_tgt_tokens

def construct_optimizer(loss, global_step, optimizer="SGD", learning_rate=1.0, clip_gradients=5.0, decay_type=tf.train.exponential_decay, decay_start_step=40000, decay_step=1000, decay_rate=0.5):
	"""Construct an optimizer to minimize loss
		Args:
			loss: the scalar need to be minimized
			global_step: the increment value
			optimizer: name of the optimizer
			clip_gradients: the gradient need clipping
			decay:
				decay_type: the tf.train.__class__ decay scheme
				decay_start_step: the step to start decay scheme
				decay_step: the step to apply the decay function
				decay_rate: the rate of decay per step
	"""
	decay_fn = lambda lr, gs: decay_type(lr, tf.maximum(gs-decay_start_step, 0), decay_step, decay_rate, name="decay_fn")
	return tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step, learning_rate=learning_rate, optimizer=optimizer, clip_gradients=clip_gradients, learning_rate_decay_fn=decay_fn, name="train_op")

class DefaultSeq2Seq:
	"""A seq2seq class that is compatible with tf estimator 
		Currently, some features are hard-coded (layer size is 2, padding is <\s>, et cetera)
	"""
	def __init__(self, vocab_files, num_units=512, params=None, **config):
		self._params = params or {}
		self.estimator = tf.estimator.Estimator(self.model_fn, config=tf.estimator.RunConfig(**config))
		self.num_units = num_units
		self.src_vocab_file, self.tgt_vocab_file = vocab_files
		self.src_vocab_size = _check_vocab_file(self.src_vocab_file)
		self.tgt_vocab_size = _check_vocab_file(self.tgt_vocab_file)
		self.dtype = tf.float32

	def verbosity(self, level):
		"""Set the tf logging verbosity level"""
		tf.logging.set_verbosity(level)

	def build_input_tensor(self, export_form=True):
		"""Build an input tensor that will handle the individual batch. Is essentially serving function
		Returns:
			tuple of (placeholder, tokens, lengths)
		"""
		tf.logging.debug("Build input in tensor form")
		with tf.variable_scope("input"):
			input_placeholder = tf.placeholder(tf.string, shape=(None, ), name="batch_input")
			input_tokenized_sparse = tf.string_split(input_placeholder)
			input_tokenized_dense = tf.sparse_tensor_to_dense(input_tokenized_sparse, name="tokenized_input", default_value="<\s>")
			# use the input string in sparse tensor form to create the input_length vector
			sparse_indices = input_tokenized_sparse.indices
			input_length_sparse = tf.SparseTensor(sparse_indices, tf.squeeze(sparse_indices[:, 1:], axis=[1]), dense_shape=input_tokenized_sparse.dense_shape)
			input_length_dense = tf.sparse_tensor_to_dense(input_length_sparse, default_value=-1)
			# input_length must have +1 since it is actually max index
			input_length = tf.reduce_max(input_length_dense, axis=1) + 1
		if(export_form):
			# features, receiver
			return tf.estimator.export.ServingInputReceiver( {
				"tokens": input_tokenized_dense, 
				"length": input_length
			}, {
				"feature": input_placeholder
			} )
		else:
			return input_placeholder, input_tokenized_dense, input_length

	def build_infer_dataset_tensor(self, infer_file):
		"""Build an input with only features for prediction
		Args:
			infer_file: the file to deal with the inference
		Returns:
			the iterator containing batched data
		"""
		batch_size = self._params.get("batch_size", 128)
		
		infer_dataset = tf.data.TextLineDataset(infer_file)
		# split string, convert to dense and drop useless first dim
		infer_dataset = infer_dataset.map( lambda item: tf.string_split([item]) )
		infer_dataset = infer_dataset.map( lambda item: tf.squeeze(tf.sparse_tensor_to_dense(item, default_value="<pad>"), axis=[0]) )
		# add the length of the tokens
		infer_dataset = infer_dataset.map( lambda item: (item, tf.size(item)))
		# batch and pad, do not change positions
		expected_shape = (tf.TensorShape([None]), tf.TensorShape([]))
		expected_filler = (EOS_TOKEN, 0)
		padding_fn = lambda dataset: dataset.padded_batch(batch_size, expected_shape, padding_values=expected_filler)
		infer_dataset = infer_dataset.apply( padding_fn )
		# apparently it still need to conform to the input_fn
		infer_dataset = tf.data.Dataset.zip( (infer_dataset, infer_dataset) )
		# map to name
		key_name = ["tokens", "length"]
		data_prepared = data_prepared.map(lambda features, labels: ({key: val for key, val in zip(key_name, features)}, None))
		# throw
		iterator = infer_dataset.make_initializable_iterator()
		tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
		return infer_dataset

	def build_batch_dataset_tensor(self, dataset_files, mode=tf.estimator.ModeKeys.TRAIN):
		"""Build an input using a combined dataset
		Args:
			dataset_files: a tuple of (src_data_file, tgt_data_file)
		Returns:
			the iterator to be called during run
			a batched set of sentences pairing ready to be processed through an iterator
		"""
		# filter away lengths above the maximum
		maximum_sentence_length = self._params.get("maximum_sentence_length", 50)
		batch_size = self._params.get("batch_size", 128)
		window_size = self._params.get("window_size", 5)
		tf.logging.debug("maximum_sentence_length: {}, batch_size: {}".format(maximum_sentence_length, batch_size))

		# read data
		src_data_file, tgt_data_file = dataset_files
		src_dataset = tf.data.TextLineDataset(src_data_file)
		tgt_dataset = tf.data.TextLineDataset(tgt_data_file)
		# zip together
		data_prepared = tf.data.Dataset.zip( (src_dataset, tgt_dataset) )
		# turn `string` into `(tokens, lengths)`
		data_prepared = _simple_processing_dataset(data_prepared)
		# the shape and padding to faciliate the bucket/batch process
		expected_inner_padded_shapes = tuple( [tf.TensorShape([None]), tf.TensorShape([])] )
		expected_padded_shapes = (expected_inner_padded_shapes, expected_inner_padded_shapes)
		expected_padded_values = ((EOS_TOKEN, 0), (EOS_TOKEN, 0))

		if(mode == tf.estimator.ModeKeys.TRAIN):
			tf.logging.debug("In mode train, will repeat and shuffle indefinitely, with dataset inside buckets and remove all sentences outside of expected range")
			# filter all outside expected range
			valid_length = lambda item: tf.logical_and(item <= maximum_sentence_length, item > 0)
			data_prepared = data_prepared.filter( lambda features, labels: tf.logical_and(valid_length(features[1]), valid_length(labels[1])) )
			# the padding function is piece of group_by_window, so need key
			padding_fn = lambda key, dataset: dataset.padded_batch(batch_size, padded_shapes=expected_padded_shapes, padding_values= expected_padded_values)
			# sort the dataset (of a sort, using group_by_window on a bucket size)
			data_prepared = data_prepared.apply(tf.contrib.data.group_by_window(key_func=lambda features, labels: tf.cast(labels[1] // window_size, dtype=tf.int64), reduce_func=padding_fn, window_size=batch_size))
			# repeat and shuffle, using very big buffer size for uniform shuffling
			data_prepared = data_prepared.apply(tf.data.experimental.shuffle_and_repeat(1000000))
		else:
			tf.logging.debug("In mode eval, only read the dataset once, with no grouping by length")
			# the padding function is used directly, so it only need dataset
			padding_fn = lambda dataset: dataset.padded_batch(batch_size, padded_shapes=expected_padded_shapes, padding_values= expected_padded_values)
			data_prepared = data_prepared.apply(padding_fn)
		# create dictionary to help with the export process, since ServingInputReceiver do not accept a tuple of Tensor
		key_name = ["tokens", "length"]
		data_prepared = data_prepared.map(lambda features, labels: ({key: val for key, val in zip(key_name, features)}, {key: val for key, val in zip(key_name, labels)}))
		iterator = data_prepared.make_initializable_iterator()
		# add to table initialization
		tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
		return data_prepared

	def build_vocab(self):
		"""Build all the necessary vocab. Splitted it here due to the fact that both build_model and compute_loss need tables at varying degree
			Returns:
				a dict that have `src_lookup_table`, `tgt_lookup_table`, `tgt_reverse_table`
		"""
		with tf.variable_scope("lookup_table"):
		# create the vocab lookup for src
			src_lookup_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=self.src_vocab_file, default_value=UNK_ID, name="src_lookup_table")
			tgt_lookup_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=self.tgt_vocab_file, default_value=UNK_ID, name="tgt_lookup_table")
			tgt_reverse_table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=self.tgt_vocab_file, default_value=UNK_TOKEN, name="tgt_reverse_lookup_table")
		return {
			"src_lookup_table": src_lookup_table, 
			"tgt_lookup_table": tgt_lookup_table, 
			"tgt_reverse_table": tgt_reverse_table
		}

	def build_model(self, features, labels, lookup_table_dict, params={}, mode=tf.estimator.ModeKeys.TRAIN):
		"""Build a model that depending on the mode will either output the logits or the predictions or both
		Args:
			features: a dict of (tokens, lengths) of tensors
			labels: a dict of (tokens, lengths) of tensors, or None
			lookup_table_dict: the result of build_vocab
			params: a dict of external parameters
			mode
		"""
		tf.logging.debug("Features tensor fed into build_model: {}".format(features))
		features_tokens = features["tokens"]
		features_length = features["length"]
		batch_size = tf.shape(features_length)[0]
		beam_size = params.get("beam_size", 10)
		max_decoder_iteration = params.get("max_decoder_iteration", 250)

		tf.logging.debug("Build embedding and create necessary values ")
		# create the embedding used
		with tf.variable_scope("embedding"):
			src_embedding = tf.get_variable("src", shape=[self.src_vocab_size, self.num_units], dtype=self.dtype)
			tgt_embedding = tf.get_variable("tgt", shape=[self.tgt_vocab_size, self.num_units], dtype=self.dtype)
		
		tf.logging.debug("Build encoder")
		with tf.variable_scope("encoder"):
			features_ids = lookup_table_dict["src_lookup_table"].lookup(features_tokens, name="features_ids")
			features_embedded = tf.nn.embedding_lookup(src_embedding, features_ids, name="features_embedded")
			forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, name="fw_cell")
			backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, name="bw_cell")
			encoder_bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, features_embedded, sequence_length=features_length, dtype=self.dtype)
			encoder_outputs = tf.concat(encoder_bi_outputs, axis=-1)

		tf.logging.debug("Build decoder using mode {}".format(mode))
		with tf.variable_scope("decoder") as decoder_scope:
			# first, the multi cell to be used
			decoder_sub_cells = [tf.nn.rnn_cell.BasicLSTMCell(self.num_units, name="cell_{:d}".format(cell_id)) for cell_id in range(1, 1+2)]
			decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_sub_cells)
			# second, the attention and its mechanism/wrapper. also adapt the encoder state to the first state of the wrapper itself
			wrapper_config = self.num_units, encoder_outputs, features_length, encoder_state
			single_cell, single_start_state, beamed_cell, beamed_start_state = _build_attention(decoder_cell, mode, batch_size=batch_size, beam_size=beam_size, wrapper_config=wrapper_config)
			# third, the projection layer to project into the tgt_vocab
			tf.logging.debug("Tgt vocab size: {:d}".format(self.tgt_vocab_size))
			projection_layer = tf.layers.Dense(self.tgt_vocab_size, name="projection_layer")

			# building the decoder for training/eval for loss
			if(mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
				tf.logging.debug("Creating logits for TRAIN/EVAL")
				tf.logging.debug("Labels tensor fed into build_model: {}".format(labels))
				labels_tokens = labels["tokens"]
				labels_length = labels["length"]
				labels_ids = lookup_table_dict["tgt_lookup_table"].lookup(labels_tokens)
				# pad the labels_ids into <s> w1 w2 ... for the input of the training helper
				# bummer, tf fill do not have dtype. Really.
				decoder_input_front_padding = tf.fill([batch_size], SOS_ID)
				decoder_input_front_padding = tf.cast( tf.expand_dims(decoder_input_front_padding, axis=1), labels_ids.dtype)
				decoder_input_length = labels_length + 1
				decoder_input_ids = tf.concat([decoder_input_front_padding, labels_ids], axis=-1, name="decode_input_ids")
				decoder_input_embedded = tf.nn.embedding_lookup(tgt_embedding, decoder_input_ids, name="decode_input_embedded")
				training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input_embedded, decoder_input_length)
				decoder = tf.contrib.seq2seq.BasicDecoder(single_cell, training_helper, single_start_state, output_layer=projection_layer)
				with tf.variable_scope("dynamic_decode") as dy_scope:
					decoder_output, final_state, decoder_length = tf.contrib.seq2seq.dynamic_decode(decoder, scope=dy_scope)
				logits = decoder_output.rnn_output
			else:
				logits = None

			# building the decoder for eval/infer for prediction
			if(mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT):
				tf.logging.debug("Creating predictions for EVAL/PREDICT")
				start_ids = tf.fill([batch_size], SOS_ID, name="decode_start_ids")
				beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=beamed_cell, embedding=tgt_embedding, start_tokens=start_ids, end_token=EOS_ID, initial_state=beamed_start_state, beam_width=beam_size, output_layer=projection_layer, reorder_tensor_arrays=True)
				with tf.variable_scope("dynamic_decode", reuse=(mode==tf.estimator.ModeKeys.EVAL)) as dy_scope:
					decoder_output, final_state, decoder_length = tf.contrib.seq2seq.dynamic_decode(beam_decoder, maximum_iterations=max_decoder_iteration, scope=dy_scope)
				assert isinstance(decoder_output, tf.contrib.seq2seq.FinalBeamSearchDecoderOutput), "Is actually {}".format(type(decoder_output))
				prediction_ids = tf.cast(decoder_output.predicted_ids, features_ids.dtype)
				# the prediction_ids is [batch, tgt_len, beam] instead of [batch, beam, tgt]. Rectify
				prediction_ids = tf.transpose(prediction_ids, perm=[0, 2, 1])
				prediction_tokens = lookup_table_dict["tgt_reverse_table"].lookup(prediction_ids)
				prediction_alignment = final_state.cell_state.alignment_history
				if(isinstance(prediction_alignment, tf.TensorArray)):
					tf.logging.warn("The alignment is still in array state, so it had not been re-gathered. Update your tensorflow version to get the correct version")
					prediction_alignment = prediction_alignment.stack(name="raw_alignment")
				# the alignment should be in [batch, beam, tgt_len, src_len] instead of currently [tgt_len, batch*beam, src_len]
				# first, transpose the tgt_len to the back. Then, reorder
				prediction_alignment = tf.transpose(prediction_alignment, perm=[1, 0, 2])
				# extra assertion to assure that the aligment is indeed in [?, tgt_len, src_len]
				with tf.control_dependencies([tf.assert_equal(tf.shape(prediction_alignment)[-1], tf.shape(features_ids)[-1]), tf.assert_equal(tf.shape(prediction_alignment)[-2], tf.shape(prediction_ids)[-1])]):
					prediction_alignment = tf.reshape(prediction_alignment, [batch_size, beam_size, tf.shape(prediction_alignment)[-2], tf.shape(prediction_alignment)[-1]])
				# replace unknown words
				prediction_tokens = _replace_unk_tokens(prediction_tokens, prediction_ids, prediction_alignment, features_tokens)
				prediction_length = final_state.lengths
				# predictions must be a dict. Bummer
				predictions = {
						"ids": prediction_ids, 
						"tokens": prediction_tokens, 
						"length": prediction_length
				}
			else:
				predictions = None

		return logits, predictions

	def compute_loss(self, logits, labels, lookup_table_dict, mode=tf.estimator.ModeKeys.TRAIN):
		"""Create a loss out of the logits and labels from the seq2seq structure
		Args:
			logits: the projected logits representing the model's prediction
			labels: the value representing the correct ids that the model should create and the correct length of each prediction
			lookup_table_dict: the lookup dictionary needed
		Returns:
			A loss as first argument and a train_op if mode is TRAIN
		"""
		# the labels must be padded at the back with an extra eos id
		labels_tokens = labels["tokens"]
		labels_length = labels["length"]
		labels_ids = lookup_table_dict["tgt_lookup_table"].lookup(labels_tokens)
		back_padding = tf.fill(tf.shape(labels_length), EOS_ID)
		back_padding = tf.cast( tf.expand_dims(back_padding, axis=1), dtype=labels_ids.dtype)
		labels_ids = tf.concat([labels_ids, back_padding], axis=-1)
		labels_length = labels_length + 1
		# the calculated entropy, masked by the extended length
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ids, name="raw_cross_entropy")
		mask = tf.sequence_mask(labels_length, maxlen=tf.shape(cross_entropy)[1], dtype=cross_entropy.dtype, name="sequence_mask")
		masked_cross_entropy = cross_entropy * mask
		# calculate the per token and per sentence loss, before sending them into the summary
		per_sentence_loss = tf.reduce_mean(tf.reduce_sum(masked_cross_entropy, axis=-1))
		per_token_loss = tf.reduce_sum(masked_cross_entropy) / tf.cast(tf.reduce_sum(labels_length), dtype=cross_entropy.dtype)
		with tf.control_dependencies([ tf.summary.scalar("sentence_loss", per_sentence_loss, family="loss"), 
																	tf.summary.scalar("token_loss", per_token_loss, family="loss")]):
			optimizer_loss = tf.identity(per_sentence_loss, name="loss")
		# create the optimizer basing on that loss
		global_step = tf.train.get_or_create_global_step()
		train_op = construct_optimizer(optimizer_loss, global_step)
		return optimizer_loss, train_op

	def model_fn(self, features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None, config=None):
		"""A model_fn that satisfies the requirement from the estimator
		Args:
			features: all features data, result of the input_fn
			labels: all labels data, likewise result of the input_fn
			mode: the mode to build the model in
			params+config: the other parameters and configurations from outside
		Returns:
			a tf.estimator.EstimatorSpec
		"""
		tf.logging.debug("Building seq2seq model")
		with tf.variable_scope("seq2seq", initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1, dtype=self.dtype)):
			lookup_table_dict = self.build_vocab()
			logits, predictions = self.build_model(features, labels, lookup_table_dict, mode=mode)
			if(mode == tf.estimator.ModeKeys.TRAIN):
				# compute the loss and create the train_op
				loss, train_op = self.compute_loss(logits, labels, lookup_table_dict, mode=mode)
			elif(mode == tf.estimator.ModeKeys.EVAL):
				loss, train_op = self.compute_loss(logits, labels, lookup_table_dict, mode=mode)
				# push the predictions and labels into a collection for hooks to access
				tf.logging.debug("Adding keys for evaluation hooks")
				labels_tokens, labels_length = labels
				prediction_tokens, prediction_length = predictions["tokens"], predictions["length"]
				tf.add_to_collection("prediction_tokens", prediction_tokens)
				tf.add_to_collection("prediction_length", prediction_length)
			elif(mode == tf.estimator.ModeKeys.PREDICT):
				# pure prediction
				loss, train_op = None, None
			elif(mode == tf.estimator.ModeKeys.EXPORT):
				raise NotImplementedError()
			else:
				raise ValueError("Mode {} invalid!".format(mode))
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, predictions=predictions)
	
	def model_hook(self, mode=tf.estimator.ModeKeys.TRAIN, eval_metric=None, eval_reference_file=None):
		"""Create and build necessary hooks to aid in the processes
			Args:
				mode: the mode that the model is running
			Returns:
				a list of SessionRunHook that can be fed into the estimator run function
		"""
		hooks = []
		if(mode == tf.estimator.ModeKeys.TRAIN):
			step_hook = tf.estimator.StepCounterHook(every_n_steps=500)
			hooks.append( step_hook )
#			logging_hook = tf.estimator.LoggingTensorHook({"per_sentence_loss": "loss"}, every_n_iter=50)
#			hooks.append( logging_hook )
		elif(mode == tf.estimator.ModeKeys.EVAL):
			if(eval_reference_file is not None):
				if(eval_metric == "bleu"):
					# use subprocess PIPE
					def script_fn(prediction_file):
						piped_file = subprocess.Popen(("cat", prediction_file), stdout=subprocess.PIPE)
						output = subprocess.check_output(("perl", "scripts/multi-bleu.perl", eval_reference_file), stdin=piped_file.stdout)
						return output
#					script = "perl scripts/multi-bleu.perl " + eval_reference_file + " < {prediction_file}"
					value_extractor = lambda result_str: ("metric/BLEU", float(re.search(r"BLEU = ([\d\.]+?),", result_str).group(1)) )
				else:
					raise ValueError("eval_metric value unrecognized: {:s}".format(eval_metric))
				metric_hook = translator_hooks.PredictionMetricHook(self.format_prediction, script_fn, model_dir=self.estimator.model_dir, summary_extraction_fn=value_extractor)
				hooks.append(metric_hook)
		return hooks

	def model_exporter(self, export_mode="best"):
		"""Create an Exporter to be used to export the models to servable versions
		Args:
			export_mode: the mode to export model on. could be best|last
		Returns:
			a tf.estimator.Exporter object for EvalSpec
		"""
		if(export_mode == "none"):
			return None
		elif(export_mode == "best"):
			return tf.estimator.BestExporter(name=export_mode, serving_input_receiver_fn=lambda: self.build_input_tensor(export_form=True))
		elif(export_mode == "latest"):
			return tf.estimator.LastExporter(name=export_mode, serving_input_receiver_fn=lambda: self.build_input_tensor(export_form=True))
		else:
			raise ValueError("Export mode {:s} invalid.".format(export_mode))
	
	def format_prediction(self, prediction, stream, n_best=1):
		"""Push the prediction tokens into proper string stream
		Args:
			prediction: the values gotten from estimator.predict, in form of (tokens, ids, length) per single arguments
			stream: the string stream for the values to print into
			n_best: the number of beam to be printed outward
		Returns:
			the satisfied stream
		"""
		tokens, _, length = prediction
		# select the n_best highest beam
		n_best = min(n_best, len(length))
		for i in range(n_best):
			tokens, length = tokens[i], length[i]
			sentence = b" ".join(tokens[:length - 1]) + b"\n"
		stream.write(sentence.decode("utf-8"))
		return stream
