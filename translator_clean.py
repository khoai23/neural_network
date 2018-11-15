import numpy as np
import sys, io, os, json, pickle, time, random
import ffBuilder as builder
import translator as legacy
import tensorflow as tf
import argparse

start_token = "<s>"
end_token = "<\s>"
unk_token = "<unk>"

def readFile(input_file_name):
	with io.open(input_file_name, "r", encoding="utf8") as input_file:
		data = input_file.readlines()
		return list(data)

def vocabListToRefDict(vocab_list, backward=False):
	if(backward):
		return {idx:word for idx, word in enumerate(vocab_list)} 
	else:
		return {word:idx for idx, word in enumerate(vocab_list)}

def addPadding(batch, src_padding_idx, tgt_padding_idx, convert_to_ndarray=True):
	src_lines, tgt_lines, src_len, tgt_len = zip(*batch)
	max_src_len = max(src_len)
	max_tgt_len = max(tgt_len)
	src_lines = [line + ([src_padding_idx] * (max_src_len - len(line))) for line in src_lines]
	tgt_lines = [line + ([tgt_padding_idx] * (max_tgt_len - len(line))) for line in tgt_lines]
#	src_lines = np.pad(np.asarray(src_lines), max_src_len, 'constant', constant_values=padding_idx)
#	tgt_lines = np.pad(np.asarray(tgt_lines), max_tgt_len, 'constant', constant_values=padding_idx)
	if(convert_to_ndarray):
		src_lines = np.array(src_lines)
		tgt_lines = np.array(tgt_lines)
	return (src_lines, tgt_lines, src_len, tgt_len)

def batchAndPad(dataset, batch_size, src_padding_idx=None, tgt_padding_idx=None):
	# TODO a bucket mode
	assert isinstance(src_padding_idx, int) and isinstance(tgt_padding_idx, int), "In batchAndPad, padding must be int"
	# sort the dataset BY tgt length THEN src length
	dataset = sorted(dataset, key=lambda x: (x[3], x[2]))
	# batch it up
	batched = []
	current_batch = []
	for item in dataset:
		if(len(current_batch) == batch_size):
			# full batch, add the padding and save it
			batched.append(addPadding(current_batch, src_padding_idx, tgt_padding_idx))
			current_batch = []
		current_batch.append(item)
	if(len(current_batch) > 0):
		# collect last item if available
		batched.append(addPadding(current_batch, src_padding_idx, tgt_padding_idx))
	return batched

def processTrainData(args, src_word_to_id, tgt_word_to_id, src_file, tgt_file, filter_fn=None):
	src_data = readFile(src_file)
	tgt_data = readFile(tgt_file)
	# filter the zipped data
	if(filter_fn is not None and callable(filter_fn)):
		zipped_data = zip(src_data, tgt_data)
		filtered_data = (x for x in zipped_data if filter_fn(x))
		src_data, tgt_data = zip(*filtered_data)
	# convert to idx
	src_unknown_idx = src_word_to_id[unk_token]
	tgt_unknown_idx = tgt_word_to_id[unk_token]
	word_to_id_fn = lambda w_dict, unk_id, line: [w_dict.get(word, unk_id) for word in line.strip().split()]
	src_data = (word_to_id_fn(src_word_to_id, src_unknown_idx, line) for line in src_data)
	tgt_data = (word_to_id_fn(tgt_word_to_id, tgt_unknown_idx, line) for line in tgt_data)
	# add end_token to tgt
	tgt_append = [ tgt_word_to_id[end_token] ]
	tgt_data = (line + tgt_append for line in tgt_data)
	
	dataset = zip(src_data, tgt_data)
	dataset_with_length = ((src_line, tgt_line, len(src_line), len(tgt_line)) for src_line, tgt_line in dataset)

	padded_data = batchAndPad(dataset_with_length, args.batch_size, src_padding_idx=src_unknown_idx, tgt_padding_idx=tgt_unknown_idx)
	return padded_data

def createSession(args):
	run_dir = args.run_dir
	mode = args.mode

	# generate vocab
	src_vocab_file = os.path.join(run_dir, "vocab." + args.src)
	tgt_vocab_file = os.path.join(run_dir, "vocab." + args.tgt)
	# generate the vocab from the files
	src_vocab = [start_token, end_token, unk_token] + legacy.getVocabFromVocabFile(src_vocab_file)
	tgt_vocab = [start_token, end_token, unk_token] + legacy.getVocabFromVocabFile(tgt_vocab_file)
	# convert to reference word_to_id dict
	src_word_to_id = vocabListToRefDict(src_vocab)
	tgt_word_to_id = vocabListToRefDict(tgt_vocab)
	
	# whatever mode, first, create the embedding and encoder
	src_embedding_fn = builder.getOrCreateEmbedding("src_embedding", create=True, vocab_size=len(src_vocab), num_units=args.num_units)
	tgt_embedding_fn = builder.getOrCreateEmbedding("tgt_embedding", create=True, vocab_size=len(tgt_vocab), num_units=args.num_units)
	# create the placeholder, embed it and create at the encoder
	src_inputs_placeholder =  tf.placeholder(dtype=tf.int32, shape=[None, None], name="src_plac")
	src_inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="src_len_plac")
	embedded_inputs = src_embedding_fn(src_inputs_placeholder)
	# the default dropout
	dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
	# encoder
	encoder_outputs, encoder_state = builder.createEncoderClean(embedded_inputs, created_dropout=dropout_placeholder)
	# create a projection layer or get existing 
	with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
		projection_layer = tf.layers.Dense(len(tgt_vocab), use_bias=False, name="decoder_projection_layer")
	# decoder, depending on the mode
	if(mode == "train"):
		tgt_inputs_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name="tgt_plac")
		tgt_inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="tgt_len_plac")
		# pad the start token to the tgt_inputs, and remove one to preserve shape similarity
		batch_size = tf.shape(tgt_inputs_placeholder)[0]
		start_token_idx = tgt_word_to_id[start_token]
		training_tgt_front_pad = tf.fill([batch_size, 1],  start_token_idx)
		decoder_inputs_ids = tf.concat([training_tgt_front_pad, tgt_inputs_placeholder], axis=1)[:, :-1]
		decoder_inputs = tgt_embedding_fn(decoder_inputs_ids)
		# run decoder, get logits
		decoder_result = builder.createDecoderClean(decoder_inputs, encoder_state, inputs_length=tgt_inputs_length_placeholder, 
				attention_mechanism=tf.contrib.seq2seq.LuongAttention, encoder_outputs=encoder_outputs, encoder_length=src_inputs_length_placeholder,
				projection_layer=projection_layer, created_dropout=dropout_placeholder, training=True, cell_size=args.num_units)
		# create train_op
		correct_decoder_outputs = tgt_inputs_placeholder
		train_loss, token_loss, train_op = builder.optimizeLossClean(decoder_result["logits"], correct_decoder_outputs, tgt_inputs_length_placeholder)
		# return the necessary tensors, placeholders and operation
		result = {
		"training_inputs": (src_inputs_placeholder, tgt_inputs_placeholder, src_inputs_length_placeholder, tgt_inputs_length_placeholder), 
		"dropout":dropout_placeholder, 
		"loss": train_loss, 
		"token_loss": token_loss,
		"train_op":train_op
		}
	elif(mode == "infer"):
		# construct the first input of start_tokens
		start_token_idx = tgt_word_to_id[start_token]
		start_tokens = tf.fill([batch_size], start_token_idx)
		end_token_idx = tgt_word_to_id[end_token]
		# run decoder, get ids/state/attention bundled in dict
		decoder_result = builder.createDecoderClean(start_tokens, encoder_state, embedding_fn=tgt_embedding_fn, end_token=end_token_idx, 
				attention_mechanism=tf.contrib.seq2seq.LuongAttention, encoder_outputs=encoder_outputs, encoder_length=src_inputs_length_placeholder, 
				projection_layer=projection_layer, training=False, cell_size=args.num_units, beam_size=args.beam_size)
		result = {
			"predict_inputs": (src_inputs_placeholder, None, src_inputs_length_placeholder, None), 
			"predictions" : decoder_result["predictions"],
			"length": decoder_result["length"],
			"log_probs": decoder_result["log_probs"],
			"alignment_history": decoder_result["alignment_history"]
		}
	else:
		raise ArgumentTypeError("Mode not valid: {:s}".format(mode))
	
	# create session and initialize
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = not args.gpu_disable_allow_growth
	session = tf.Session(config=config)
	session.run(tf.global_variables_initializer())
	# check for saved values
	builder.loadFromPath(session, args.save_path)
	
	result["session"] = session
	result["src_word_to_id"] = src_word_to_id
	result["tgt_word_to_id"] = tgt_word_to_id
	return result

def trainSession(session_data, current_batch, dropout=None):
	session = session_data["session"]
	training_placeholders = session_data["training_inputs"]
	# zip the placeholder with the data
	bundled = zip(training_placeholders, current_batch)
	feed_dict = {k:v for k, v in bundled}
	if(dropout and dropout < 1.0):
		dropout_pl = session_data["dropout"]
		feed_dict[dropout_pl] = dropout
	all_train_ops = session_data["loss"], session_data["token_loss"], session_data["train_op"]
	loss, token_loss, _ = session.run(all_train_ops, feed_dict=feed_dict)
	return loss, token_loss

if __name__ == "__main__":
	# Fix later
	parser = argparse.ArgumentParser(description='Run RNN.')
	args = parser.parse_args([])
	args.mode = "train"
	args.run_dir = sys.argv[1] #/home/quan/Workspace/Data/iwslt15
	args.file_name = "train"
	args.src = "en"
	args.tgt = "vi"
	args.num_units = 512
	args.batch_size = 128
	args.beam_size = 5
	args.gpu_disable_allow_growth = False
	args.save_path = "./run_trash.sav"
	args.manual_load_train_data = True
	args.epoch = 10
	args.dropout = 0.8

	filter_fn = lambda x: len(x[0].strip().split()) <= 50 and len(x[1].strip().split()) <= 50
	
	timer = time.time()
	session_data = createSession(args)
	print("Session created in {:.2f}s".format(time.time() - timer))
	
	if(args.mode == "train"):
		src_word_to_id = session_data["src_word_to_id"]
		tgt_word_to_id = session_data["tgt_word_to_id"]
		dump_file = os.path.join(args.run_dir, "train.batch")
		if(not os.path.isfile(dump_file) or args.manual_load_train_data):
			print("Create data in training...")
			with io.open(dump_file, "wb") as dump_file:
				src_file = os.path.join(args.run_dir, "train." + args.src)
				tgt_file = os.path.join(args.run_dir, "train." + args.tgt)
				batched_dataset = processTrainData(args, src_word_to_id, tgt_word_to_id, src_file, tgt_file, filter_fn=filter_fn)
				pickle.dump(batched_dataset, dump_file)
		else:
			with io.open(dump_file, "rb") as dump_file:
				batched_dataset = pickle.load(dump_file)
		# print(batched_dataset)
		print("Dataset: {:d} batches".format(len(batched_dataset)))
		indexed_batches = list(enumerate(batched_dataset))
		timer = time.time()
		for ep in range(args.epoch):
			# shuffle each epoch
			print("Starting epoch {:d}".format(ep))
			epoch_timer = time.time()
			random.shuffle(indexed_batches)
			for batch_idx, batch in indexed_batches:
				loss, token_loss = trainSession(session_data, batch, dropout=args.dropout)
				print("Loss {:.2f}({:.2f}) for batch {:d}".format(loss, token_loss, batch_idx))
			builder.saveToPath(session_data["session"], args.save_path)
			print("End epoch {:d}, time passed {:.2f}s".format(ep, time.time() - epoch_timer))
		print("End training, time passed {:.2f}s".format(time.time() - timer))
	else:
		raise NotImplementedError("Not coded!")

