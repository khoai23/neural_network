import tensorflow as tf
import numpy as np
from ffBuilder import createRandomArray
import sys
# import helpers

EMBEDDING_SIZE = 4
EMBEDDING_NUM = 16
HIDDEN_UNIT_SIZE = 16
PAD = 0
EOS = 1

RUN_DUMMY_DATA = False

# reset graph and session
tf.reset_default_graph()
session = tf.get_default_session()
if(session is not None):
	session.close()
''' Initialize a new session and draw needed graph '''
session = tf.Session()
# embedding
embeddings = tf.Variable(initial_value=tf.random_uniform([EMBEDDING_NUM, EMBEDDING_SIZE], -1.0, 1.0, dtype=tf.float32))
# create input tensor, dimension [inputLength, batchSize]
encoderInput = tf.placeholder(shape=[None, None], dtype=tf.int32)
# create lookups from embedding to input
inputLookup = tf.nn.embedding_lookup(embeddings, encoderInput)
### tensorCurrentShape = tf.shape(inputLookup)
# encoder_inputs_embedded = tf.reshape(inputLookup, [-1, -1, EMBEDDING_SIZE])
# create encoder and assign the inputLookup into it
encoder_cell = tf.contrib.rnn.LSTMCell(HIDDEN_UNIT_SIZE)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, inputLookup, dtype=tf.float32, time_major=True,)

# decoder will also use the embedding specified above for its input and output, with the same complication
output = tf.placeholder(shape=[None, None], dtype=tf.int32)
decoderInput = tf.placeholder(shape=[None, None], dtype=tf.int32)
decoderInputLookup = tf.nn.embedding_lookup(embeddings, decoderInput)
### tensorCurrentShape = tf.shape(outputLookup)
# decoder_inputs_embedded = tf.reshape(outputLookup, [-1, -1, EMBEDDING_SIZE])
# create decoder and link it with the hidden size of the encoder
decoder_cell = tf.contrib.rnn.LSTMCell(HIDDEN_UNIT_SIZE)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoderInputLookup, initial_state=encoder_final_state,
	dtype=tf.float32, time_major=True, scope="plain_decoder",)
# logits for prediction/training, prediction being the idx gotten from the logits layer
decoder_logits = tf.contrib.layers.linear(decoder_outputs, EMBEDDING_NUM)
# prediction is the largest value among the logits layer, inaccessible 
decoder_prediction = tf.argmax(decoder_logits, 2)
# output is converted from int to onehot vector as the result of the decoder
outputOneHot = tf.one_hot(output, depth=EMBEDDING_NUM, dtype=tf.float32)
# the result of decoder logits are the softmax distribution of word probability
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=outputOneHot,logits=decoder_logits,)
# custom checker - difference between decoderlogits in softmax and onehot result\
decoder_logits = tf.nn.softmax(decoder_logits)
difference = outputOneHot - decoder_logits
# loss function and optimizer(learner)
loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

session.run(tf.global_variables_initializer())

# Helper function
# batch will transpose to fit into time-major, which defeat the purpose of using time-major mode in the first place...
def batch(inputs, max_sequence_length=None):
	"""
	Args:
		inputs:
			list of sentences (integer lists)
		max_sequence_length:
			integer specifying how large should `max_time` dimension be.
			If None, maximum sequence length would be used

	Outputs:
		inputs_time_major:
			input sentences transformed into time-major matrix 
			(shape [max_time, batch_size]) padded with 0s
		sequence_lengths:
			batch-sized list of integers specifying amount of active 
			time steps in each input sequence
	"""

	sequence_lengths = [len(seq) for seq in inputs]
	batch_size = len(inputs)

	if max_sequence_length is None:
		max_sequence_length = max(sequence_lengths)

	inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD

	for i, seq in enumerate(inputs):
		for j, element in enumerate(seq):
			inputs_batch_major[i, j] = element

	# [batch_size, max_time] -> [max_time, batch_size]
	inputs_time_major = inputs_batch_major.swapaxes(0, 1)

	return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
					 vocab_lower, vocab_upper,
					 batch_size):
	""" Generates batches of random integer sequences,
		sequence length in [length_from, length_to],
		vocabulary in [vocab_lower, vocab_upper]
	"""
	if length_from > length_to:
			raise ValueError('length_from > length_to')
	
	def random_length():
		if length_from == length_to:
			return length_from
		return np.random.randint(length_from, length_to + 1)
	
	while True:
		yield [
			np.random.randint(low=vocab_lower,
							  high=vocab_upper,
							  size=random_length()).tolist()
			for _ in range(batch_size)]

if(RUN_DUMMY_DATA):
	''' Now, run graph on dummy data. This one will generate completely asinine result, as it is untrained'''
	batch_ = [[6], [3, 4], [9, 8, 7]]

	batch_, batch_length_ = batch(batch_)
	print('batch_encoded:\n' + str(batch_))

	din_, dlen_ = batch(np.ones(shape=(3, 1), dtype=np.int32),
								max_sequence_length=4)
	print('decoder inputs:\n' + str(din_))

	pred_ = session.run(decoder_prediction,
		feed_dict={ encoderInput: batch_, decoderInput: din_, })
	print('decoder predictions:\n' + str(pred_))
	
	sys.exit()

''' Run the actual training on a valid variable graph'''
# Create random batch to train
trainingSize = 64
batches = random_sequences(length_from=3, length_to=8,
								   vocab_lower=EOS, vocab_upper=EMBEDDING_NUM,
								   batch_size=trainingSize)
print('head of the batch:')
for seq in next(batches)[:10]:
	print(seq)

# get the data from the batches generator, and transpose it again
def next_feed():
	_batch = next(batches)
	encoder_inputs_, _ = batch(_batch)
	decoder_targets_, _ = batch(
		[(sequence) + [EOS] for sequence in _batch]
	)
	decoder_inputs_, _ = batch(
		[[EOS] + (sequence) for sequence in _batch]
	)
	#print("NEXTFEED: ", _batch)
	#print("toInput", encoder_inputs_)
	return {
		encoderInput: encoder_inputs_,
		decoderInput: decoder_inputs_,
		output: decoder_targets_,
	}

loss_track = []
max_batches = 3001
batches_in_epoch = 1000

try:
	for _batch in range(max_batches):
		feed = next_feed()
		_, l = session.run([train_op, loss], feed)
		loss_track.append(l)
		
		if _batch == 0 or _batch % batches_in_epoch == 0:
			print('batch {}'.format(_batch))
			print('  minibatch loss: {}, minibatch cross_entrophy later'.format(l))
			predict_, difference_, resultInSoftmax, correctOnehot = session.run([decoder_prediction, difference, outputOneHot, decoder_logits], feed)
			
			#print( np.transpose(feed[encoderInput])[4], '->', np.transpose(feed[decoderInput])[4], '->', np.transpose(predict_)[4], '=', np.transpose(feed[output])[4])
			#for item in zip(resultInSoftmax[4], correctOnehot[4], difference_[4]):
			#	print("{} - {} = {}".format(item[1], item[0], item[2]))
			
			for i, (inp, pred) in enumerate(zip(np.transpose(feed[encoderInput]),np.transpose(predict_))):
				print('  sample {}:'.format(i + 1))
				print(' input/output > {}'.format(inp))
				# print('    output    > {}'.format(out))
				print('  predicted   > {}'.format(pred))
				if i >= 2:
					break
	#print(loss_track)
except KeyboardInterrupt:
    print('training interrupted')