import tensorflow as tf

class BiRNNLanguageModel:
	def __init__(self, vocabulary_size, embedding_size=128, num_layers=2, num_hidden=None):
		self.embedding_size = embedding_size
		self.num_layers = num_layers
		self.num_hidden = num_hidden if num_hidden is not None else embedding_size

		self.x = tf.placeholder(tf.int32, [None, None])
		self.keep_prob = tf.placeholder_with_default(1.0, shape=())
		self.batch_size = tf.shape(self.x)[0]

		self.lm_input = self.x
		self.lm_output = self.x[:, 1:-1]
		# intriguing. padding is 0 so only positive values are expected
		self.seq_len = tf.reduce_sum(tf.sign(self.lm_input), 1)

		with tf.name_scope("embedding"):
			init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
			embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
			lm_input_emb = tf.nn.embedding_lookup(embeddings, self.lm_input)

		with tf.name_scope("bi-rnn"):
			def make_cell():
				cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)
				cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
				return cell

			fw_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
			bw_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
			rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
				fw_cell, bw_cell, lm_input_emb, sequence_length=self.seq_len, dtype=tf.float32)

			fw_outputs = rnn_outputs[0][:, :-2, :]
			bw_outputs = rnn_outputs[1][:, 2:, :]
			merged_output = tf.concat([fw_outputs, bw_outputs], axis=2)

		with tf.name_scope("output"):
			self.logits = tf.layers.dense(merged_output, vocabulary_size)

	def prepare_training(self, optimizer="Adam", learning_rate=1e-3, dropout=0.2):
		global_step = tf.train.get_or_create_global_step()
		self.loss = loss = tf.contrib.seq2seq.sequence_loss(
				logits=self.logits,
				targets=self.lm_output,
				weights=tf.sequence_mask(self.seq_len - 2, tf.shape(self.x)[1] - 2, dtype=tf.float32),
				average_across_timesteps=True,
				average_across_batch=True,
				name="loss"
			)
		self.train_op = train_op = tf.contrib.layers.optimize_loss(
				loss, 
				global_step,
				learning_rate,
				optimizer,
				clip_gradients=5.0,
				name="train_op"
			)
		self.dropout = dropout
		return loss, train_op

	def train(self, session, train_data):
		feed_dict = {self.lm_input:train_data, self.keep_prob:1.0-self.dropout}
		return session.run([self.loss, self.train_op], feed_dict=feed_dict)
