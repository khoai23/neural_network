import tensorflow as tf
import ffBuilder as builder
import numpy as np
import sys, io, os, time
import json, argparse
from xer import levenshtein
from random import shuffle
from itertools import cycle

def batch(iterable, n=1):
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]

def readPretrainedEmbedding(file_path):
	with io.open(file_path, "r", encoding="utf8") as emb_file:
		lines = emb_file.readlines()
		embedding_array = []
		word_list = []
		# comprehend the file
		for idx, line in enumerate(lines):
			if(idx == 0 and len(line.strip().split()) == 2):
				# first line is glove format, ignore
				print("Ignore first line: {:s}".format(line))
				continue
			word, vector = line.strip().split(" ", 1)
			word_list.append(word)
			embedding_array.append(np.fromstring(vector, dtype=float, sep=" "))
		# convert the embedding_array to ndarray
		embedding_array = np.asarray(embedding_array)
		print("Read word list {} items, embeddings shape {}".format(len(word_list), np.shape(embedding_array)))
		return word_list, embedding_array

def readJSONData(file_path, input_field="input", output_field="output"):
	assert os.path.isfile(file_path), "File {:s} not valid!".format(file_path)
	# json_data = json.loads(file_path, encoding="utf8")
	with io.open(file_path, "r", encoding="utf8") as json_file:
		json_data = list(json.load(json_file))
	full_data_list = []
	for block in json_data:
		# data is tuple of (input, correct_output)
		# assert input_field in block and output_field in block, "Block error: {}".format(block)
		inputs_batch = block[input_field]
		output_batch = block[output_field]
		full_data_list.extend(zip(inputs_batch, output_batch))
	return full_data_list

def score(filter_type, l1, l2):
	assert filter_type == "wer" or filter_type == "cer"
	# Compare l1, l2 by filter type
	if(filter_type == "wer"):
		l1, l2 = l1.split(), l2.split()
	_, (s, i, d) = levenshtein(l1, l2)
	return 1.0 - ( (s + i + d) / max(len(l1), len(l2)) )

def filterData(data, filter_threshold=0.9, score_fn=None):
	# filter duplicate basing on wer
	assert score_fn is not None, "Filter must have a callable score_fn"
	filtered_item_list = []
	for idx, item in enumerate(data):
		# check with all passed
		checker = filtered_item_list
		duplicate_found = False
		for check_idx, check_item in enumerate(checker):
			# comparing inputs
			score = score_fn(item[0], check_item[0])
			if(score >= filter_threshold):
				print("Duplicate items @idx {:d}-{:d}, score {:.2f}, items {:s}-{:s}".format(idx, check_idx, score * 100.0, item[0], check_item[0]))
				duplicate_found = True
				break
		if(not duplicate_found):
			# no duplication, add to filtered items
			filtered_item_list.append(item)
		if(idx >= 1000):
			break
	return filtered_item_list

def constructParser():
	parser = argparse.ArgumentParser(description='A rewrite of sentiment analysis.')
	parser.add_argument('-m','--mode', type=str, default='default', choices=["default"], help='Mode of the parser.')
	parser.add_argument('--data_dir', type=str, default="./", help='Location of files to train, default ./')
	parser.add_argument('--gpu_disable_allow_growth', action="store_true", help='Use to disable gpu_allow_growth')
	return parser.parse_args()

def sentimentAttentionRNN(args):
	# session
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = not args.gpu_disable_allow_growth
	session = tf.Session(config=config)
	# INPUTS 
	# embedding path
	words, embeddings = readPretrainedEmbedding(args.embedding_file)
	default_word = args.default_word
	default_idx = words.find(default_word)
	# initiate entrance point and string_to_id func
	input_placeholder = tf.placeholder(tf.string, shape=(None, ), name="batch_input")
	word_indices, word_strings = zip(*enumerate(words))
	word_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(word_strings, word_indices), default_idx, name="word_to_id_table")
	# tokenize and feed through table
	input_tokenized_sparse = tf.string_split(input_placeholder)
	input_tokenized_dense = tf.sparse_tensor_to_dense(input_tokenized_sparse, name="tokenized_input", default_value="<\s>")
	# each indices in length is its second axis, which help with 
	sparse_indices = input_tokenized_sparse.indices
	input_length_sparse = tf.SparseTensor(sparse_indices, tf.squeeze(sparse_indices[:, 1:], axis=[1]), dense_shape=input_tokenized_sparse.dense_shape)
	input_length_dense = tf.sparse_tensor_to_dense(input_length_sparse, default_value=-1)
	input_length = tf.reduce_max(input_length_dense, axis=0)
	# lookup the ids by the table
	input_indices = word_table.lookup(input_tokenized_dense, name="ids_input")
	# create embedding and lookup the indices
	embeddings_tensor = tf.constant(embeddings, dtype=tf.float32, name="embedding_tensor")
	inputs = tf.nn.embedding_lookup(embeddings_tensor, input_indices)
	# RESULTS
	result_placeholder = tf.placeholder(tf.int32, shape=(None, ), name="batch_result")
	result_sigmoid = tf.minimum(tf.cast(result_placeholder, tf.float32) - 1.0, 4.0) / 4.0
	# MISC
	dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
	# run bidirectional RNN 
	outputs, last_state = builder.createEncoderClean(inputs, cell_size=args.cell_size, created_dropout=dropout_placeholder)
	attention_base = tf.get_variable("attention_base", shape=[1, args.cell_size * 2, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
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
	mask_choice = tf.Print(mask_choice, [tf.shape(mask_choice), tf.shape(attention_unmasked)])
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
	predictions = tf.nn.sigmoid(predictions_raw)
	# create train_op
	global_step = tf.train.get_or_create_global_step()
	learning_rate = 1.0
	optimizer = "SGD"
	train_op = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, name="train_op")
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	return session, (input_placeholder, result_placeholder, dropout_placeholder), (loss, train_op), predictions

if __name__ == "__main__":
	args = constructParser()
	args.data_dir = "../../Data/espresso"
	args.embedding_file = "cc.vi.300.vec"
	args.embedding_file = os.path.join(args.data_dir, args.embedding_file)
	args.data_file = "ttn_data.json_2018-11-12"
	args.data_file = os.path.join(args.data_dir, args.data_file)
	args.save_file = "test.sav"
	args.save_file = os.path.join(args.data_dir, args.save_file)
	args.cell_size = 512
	args.filter_type = "wer"
	args.epoch = 10
	args.batch_size = 64
	args.dropout = 0.8
	args.filter_range = 1000
	args.default_word = "unknown"
	
	session, placeholders, trainers, predictions = sentimentAttentionRNN(args)
	print("Build session done.")
	builder.loadFromPath(session, args.save_file)
	score_fn = lambda l1, l2: score(args.filter_type, l1, l2)
	dump_file_path = os.path.join(args.data_dir, args.data_file + "_filtered")
	if(not os.path.isfile(dump_file_path)):
		print("File path without dump, create and filter data and dump it")
		full_data = readJSONData(args.data_file, input_field="ttn_bl_noi_dung", output_field="ttn_bl_diem")
		print("Load data complete, data length {:d}".format(len(full_data)) )
		filtered_data = filterData(full_data, score_fn=score_fn, filter_threshold=0.9)
		with io.open(dump_file_path, "w") as dump_file:
			json.dump(filtered_data, dump_file)
			print("Filter data dumped to {:s}".format(dump_file_path))
	else:
		print("File path with dump, load it from {:s}".format(dump_file_path))
		with io.open(dump_file_path, "r") as dump_file:
			filtered_data = json.load(dump_file)
	
	# shuffle and create eval/train set
	shuffle(filtered_data)
	eval_set_size = min(1000, len(filtered_data) // 10)
	train_set_size = len(filtered_data) - eval_set_size
	print("Set size: train {:d}, eval {:d}".format(train_set_size, eval_set_size))
	train_set_unordered, eval_set = filtered_data[:train_set_size], filtered_data[train_set_size:]
	# order the set by length
	train_set_ordered = list(sorted(train_set_unordered, key=lambda item: len(item[0].split())))
	training_dropout = args.dropout
	input_pl, result_pl, dropout_pl = placeholders
	loss, train_op = trainers
	timer = time.time()
	for i in range(args.epoch):
		# start training
		batch_iters = batch(train_set_ordered, n=args.batch_size)
		for batch_idx, batch in enumerate(batch_iters):
			inputs, correct_outputs = zip(*batch)
			inputs = list(inputs)
			correct_outputs = [int(o) for o in correct_outputs]
			loss, _ = session.run(trainers, feed_dict={input_pl:inputs, result_pl:correct_outputs, dropout_pl:training_dropout})
			print("Iter {:d}, batch {:d}, loss {:.4f}, time passed {:.2f}".format(i, batch_idx, loss, time.time() - timer))
		# start eval
		eval_inputs, eval_results = zip(*eval_set)
		eval_predictions = session.run(predictions, feed_dict={input_pl:list(eval_inputs)})
		eval_full_result = zip(eval_inputs, eval_predictions, eval_results)
		for item in eval_full_result:
			print("Evaluation: sentence {:s}, prediction(unscaled) {:.4f}, correct {:s}".format(eval_full_result))
	builder.saveToPath(session, args.save_file)
