import tensorflow as tf
import os, abc
import argparse
from translator_module import construct_optimizer

UNK_TOKEN = "<unk>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "<\s>"
UNK_ID = 0

WORD_FILE_LOCATION = "/home/quan/Workspace/Source/translator/txt/Viet74K.txt"
WORD_RELATION_DUMP = "/home/quan/Workspace/Source/translator/txt/xer_distance.txt"

class LanguageModel:
	@abc.abstractmethod
	def model_fn(self):
		raise NotImplementedError()
		
	@abc.abstractmethod
	def build_model(self):
		raise NotImplementedError()
	
	@abc.abstractmethod
	def compute_loss(self):
		raise NotImplementedError()

class BidirLanguageModel(LanguageModel):
	def __init__(self, vocab_file, num_units=512, num_layers=2, params=None, **config):
		self._params = params or {}
		self._num_units = num_units
		self._num_layers = num_layers
		self.estimator = tf.estimator.Estimator(self.model_fn, config=tf.estimator.RunConfig(**config))
		self._vocab_file = vocab_file
		self.dtype = tf.float32

	def build_input(self, mode=tf.estimator.ModeKeys.TRAIN, input_files=None, maximum_sentence_length=100):
		"""Build the inputs for the model. All three modes have roughly the same format, except train which has shuffle and repeat"""
		dataset = tf.data.TextLineDataset(input_files)
		# convert to tensor of tokens
		dataset = dataset.map(lambda line: tf.string_split([line]))
		force_to_dense = lambda item: (tf.squeeze(tf.sparse_tensor_to_dense(item[0], default_value="<pad>"), axis=[0]), item[1])
		dataset = dataset.map(force_to_dense)

		# pad the tokens with the eos/sos tokens and length
		front_pad, back_pad = tf.convert_to_tensor([SOS_TOKEN]), tf.convert_to_tensor([EOS_TOKEN])
		dataset = dataset.map(lambda item: (item, tf.concat([front_pad, item, back_pad]), tf.size(item)))
		
		# filter length and batch by length if TRAIN, otherwise do not swap positions
		expected_shape = tuple( [tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])] )
		expected_filler = ("<pad>", "<pad>", 0)
		batch_size = self._params.get("batch_size", 128)
		if(mode == tf.estimator.ModeKeys.TRAIN):
			tf.logging.debug("In train, executing batching/filtering with length")
			window_size = 5
			# filter length
			dataset = dataset.filter(lambda item: item[-1] <= maximum_sentence_length)
			# perform bucket batch
			padding_fn = lambda key, dataset: dataset.padded_batch(batch_size, padded_shapes=expected_shape, padding_values=expected_filler)
			dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=lambda features: tf.cast(features[-1] // window_size, dtype=tf.int64), reduce_func=padding_fn, window_size=batch_size))
			# shuffle and repeat
			dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(1000000))
		else:
			tf.logging.debug("In eval/predict, batch using normal indices")
			padding_fn = lambda d: d.padded_batch(batch_size, padded_shapes=expected_shape, padding_values=expected_filler)
			dataset = dataset.apply(padding_fn)
		# regardless of mode, convert the tuple into dict
		key_name = ["tokens", "padded_tokens", "length"]
		dataset = dataset.map(lambda item: {k: val for k, v in zip(key_name, item)})
		iterator = dataset.make_initializable_iterator()
		# add to table initialization
		tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
		return dataset

	def build_vocab(self):
		"""Build a tuple of (word_to_ids_table, ids_to_words_table)"""
		# maybe create vocab or replace, whatever
		with tf.variable_scope("lookup_table"):
		# create the vocab lookup for src
			src_lookup_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=self._vocab_file, default_value=UNK_ID, name="src_lookup_table")
			src_reverse_table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=self._vocab_file, default_value=UNK_TOKEN, name="tgt_reverse_lookup_table")
		return src_lookup_table, src_reverse_table

	def model_fn(self, features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None, config=None):
		"""A model_fn that satisfies the requirement from the estimator
		Args:
			features: all features data, result of the input_fn
			labels: either True or None depending on the modes
			mode: the mode to build the model in
			params+config: the other parameters and configurations from outside
		Returns:
			a tf.estimator.EstimatorSpec
		"""
		with tf.variable_scope("language_model", initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1, dtype=self.dtype)):
			lookup_tables = self.build_vocab()
			projection = self.build_model(features, labels, lookup_tables, params=params, mode=mode)
			if(mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
				loss, train_op = self.compute_loss(projection, features, lookup_tables, params=params, mode=mode)
				predictions = None
			elif(mode == tf.estimator.ModeKeys.INFER):
				loss, train_op = None, None
				trimmed_projection = self.trim_projections(projection, trim_size=20)
				predictions = {"raw": projection, "scores": trimmed_projection, "length": features["length"]}

		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, predictions=predictions)

	def build_model(self, features, labels, lookup_tables, params={}, mode=tf.estimator.ModeKeys.TRAIN):
		"""Build a model using valid monolingual sentences
			Args:
				features: the batched sentences
				labels: either True or None depending on the modes
				lookup_tables: tuple of (tokens_to_ids, ids_to_tokens)
				params: additional parameters
				mode: the tf.estimator.ModeKeys
		"""
		tf.logging.debug("Features dict received: {}".format(features))
		tokens = features["padded_tokens"]
		length = features["length"]
		# construct the bidirectional

		tf.logging.debug("Build embedding and create necessary values ")
		# create the embedding used
		with tf.variable_scope("embedding"):
			src_embedding = tf.get_variable("values", shape=[self.src_vocab_size, self.num_units], dtype=self.dtype)

		word_to_ids_table, _ = lookup_tables
		tf.logging.debug("Build bidirectional access")
		with tf.variable_scope("rnn"):
			features_ids = word_to_ids_table.lookup(features_tokens, name="features_ids")
			features_embedded = tf.nn.embedding_lookup(src_embedding, features_ids, name="features_embedded")
			forward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, name="fw_cell")
			backward_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, name="bw_cell")
			encoder_bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, features_embedded, sequence_length=features_length, dtype=self.dtype)
		
		# concat the forward and backward and fed it into projection
		with tf.variable_scope("projection"):
			fw_output, bw_output = encoder_bi_outputs
			logits = tf.concat([fw_output[:, :-2, :], bw_output[:, 2:, :]], axis=-1)
			projection_layer = tf.layers.Dense(self.tgt_vocab_size, name="projection_layer")
			projection = projection_layer(logits)

		return projection

	def compute_loss(self, projection, labels, lookup_tables, params={}, mode=tf.estimator.ModeKeys.TRAIN):
		"""Compute the losses from the correct labels
			Args:
				projection: the result from build_model
				labels: the correct tokens and length
				lookup_tables: tuple of (tokens_to_ids, ids_to_tokens)
				mode: the tf.estimator.ModeKeys
		"""
		labels_tokens = labels["tokens"]
		labels_length = labels["length"]
		labels_smoothing = params.get("labels_smoothing", 0.6)
		# convert to ids
		_, ids_to_words_table = lookup_tables
		# create the smoothed labels
		vocab_size = tf.shape(projection)[-1]
		smoothing_off_value = (1.0 - labels_smoothing) / tf.cast(vocab_size - 1, labels_smoothing)
		smoothed_labels = tf.one_hot(labels_tokens, vocab_size, on_value=labels_smoothing, off_value=smoothing_off_value, name="smoothed_labels")
		# calculate the cross entropy
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=smoothed_labels, logits=projection, name="cross_entropy")
		# create the mask using labels_length
		mask = tf.sequence_mask(labels_length, dtype=cross_entropy.dtype, name="mask")
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
	
	def trim_projections(self, projection, trim_size=20):
		"""Trim the projection down to {trim_size} most probable tokens per position """
		return tf.nn.top_k(projection, k=trim_size, name="trimmed_projection")[0]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Language Models.')
	parser.add_argument('mode', type=str, choices=["train", "infer", "score"], help="?")
	parser.add_argument('-m', '--model_type', type=str, choices=["bidir"], default="bidir", help='The type of model used')
	parser.add_argument('--data', type=str, required=True, help="Location of monolingual data")
	parser.add_argument('--size', type=int, default=128, help="Size of hidden/embeddings within model")
	args = parser.parse_args()
	
	# process data
	assert os.path.isfile(args.data)
