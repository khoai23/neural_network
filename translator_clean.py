import numpy as np
import io, os, pickle, time, random
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
	indexed_dataset = enumerate(dataset)
	sorted_with_idx = sorted(indexed_dataset, key= lambda x: (x[1][3], x[1][2]))
	# dataset = sorted(dataset, key=lambda x: (x[3], x[2]))
	indices, dataset = zip(*sorted_with_idx)
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
	return batched, indices

def processData(args, src_word_to_id, tgt_word_to_id, src_file, tgt_file=None, filter_fn=None):
	src_data = readFile(src_file)
	if(tgt_file is None):
		print("No tgt_file specified, using dummy.")
		tgt_data = ["" for _ in src_data]
	else:
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

	padded_data, indices = batchAndPad(dataset_with_length, args.batch_size, src_padding_idx=src_unknown_idx, tgt_padding_idx=tgt_unknown_idx)
	return padded_data, indices

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
	batch_size = tf.shape(src_inputs_placeholder)[0]
	embedded_inputs = src_embedding_fn(src_inputs_placeholder)
	# the default dropout
	dropout_placeholder = tf.placeholder_with_default(1.0, shape=(), name="dropout")
	# encoder
	encoder_outputs, encoder_state = builder.createEncoderClean(embedded_inputs, created_dropout=dropout_placeholder)
	# decoder, depending on the mode
	if(mode == "train"):
		tgt_inputs_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name="tgt_plac")
		tgt_inputs_length_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="tgt_len_plac")
		# pad the start token to the tgt_inputs, and remove one to preserve shape similarity
		start_token_idx = tgt_word_to_id[start_token]
		training_tgt_front_pad = tf.fill([batch_size, 1],  start_token_idx)
		decoder_inputs_ids = tf.concat([training_tgt_front_pad, tgt_inputs_placeholder], axis=1)[:, :-1]
		decoder_inputs = tgt_embedding_fn(decoder_inputs_ids)
		# run decoder, get logits
		decoder_result = builder.createDecoderClean(decoder_inputs, encoder_state, len(tgt_vocab), inputs_length=tgt_inputs_length_placeholder, 
				attention_mechanism=tf.contrib.seq2seq.LuongAttention, encoder_outputs=encoder_outputs, encoder_length=src_inputs_length_placeholder,
				created_dropout=dropout_placeholder, training=True, cell_size=args.num_units)
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
		decoder_result = builder.createDecoderClean(start_tokens, encoder_state, len(tgt_vocab), embedding_fn=tgt_embedding_fn, end_token=end_token_idx, 
				attention_mechanism=tf.contrib.seq2seq.LuongAttention, encoder_outputs=encoder_outputs, encoder_length=src_inputs_length_placeholder, 
				training=False, cell_size=args.num_units, beam_size=args.beam_size)
		result = {
			"predict_inputs": (src_inputs_placeholder, None, src_inputs_length_placeholder, None), 
			"predictions" : decoder_result["predictions"],
			"length": decoder_result["length"],
			"log_probs": decoder_result["log_probs"],
			"alignment_history": decoder_result["alignment_history"]
		}
	else:
		raise argparse.ArgumentTypeError("Mode not valid: {:s}".format(mode))
	
	# create session and initialize
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = not args.gpu_disable_allow_growth
	session = tf.Session(config=config)
	session.run(tf.global_variables_initializer())
#	print([t.name for t in tf.trainable_variables()])
	# check for saved values
	builder.loadFromPath(session, args.save_path, debug=True)
	
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

def inferSession(session_data, current_batch):
	session = session_data["session"]
	predict_placeholders = session_data["predict_inputs"]
	bundled = zip(predict_placeholders, current_batch)
	feed_dict = {k:v for k, v in bundled if k is not None}
	all_pred_ops = session_data["predictions"], session_data["length"], session_data["log_probs"]
	predictions, predictions_length, _ = session.run(all_pred_ops, feed_dict=feed_dict)
	return predictions, predictions_length

def getGlobalStep(session):
	session_global_step = tf.train.get_global_step()
	return session.run(session_global_step)

if __name__ == "__main__":
	# Fix later
	parser = argparse.ArgumentParser(description='Run RNN.')
	parser.add_argument('-m', '--mode', choices=["train", "infer", "eval", "export"], help="Mode of the program")
	args = parser.parse_args()
	args.run_dir = "/home/quan/Workspace/Data/iwslt15"
	args.file_name = "tst2012"
	args.src = "en"
	args.tgt = "vi"
	args.num_units = 512
	args.batch_size = 128
	args.beam_size = 10
	args.gpu_disable_allow_growth = False
	args.save_path = "./data/run_trash.sav"
	args.manual_load_train_data = False
	args.epoch = 9
	args.dropout = 0.8

	
	timer = time.time()
	session_data = createSession(args)
	print("Session created in {:.2f}s".format(time.time() - timer))
	src_word_to_id = session_data["src_word_to_id"]
	tgt_word_to_id = session_data["tgt_word_to_id"]
	
	if(args.mode == "train"):
		filter_fn = lambda x: 0 < len(x[0].strip().split()) <= 50 and 0 < len(x[1].strip().split()) <= 50
		dump_file = os.path.join(args.run_dir, "train.batch")
		if(not os.path.isfile(dump_file) or args.manual_load_train_data):
			print("Create data in training...")
			with io.open(dump_file, "wb") as dump_file:
				src_file = os.path.join(args.run_dir, "train." + args.src)
				tgt_file = os.path.join(args.run_dir, "train." + args.tgt)
				# training data won't need sentences indices
				batched_dataset, _ = processData(args, src_word_to_id, tgt_word_to_id, src_file, tgt_file=tgt_file, filter_fn=filter_fn)
				pickle.dump(batched_dataset, dump_file)
		else:
			print("Load processed data...")
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
			builder.saveToPath(session_data["session"], args.save_path, getGlobalStep(session_data["session"]))
			print("End epoch {:d}, time passed {:.2f}s".format(ep, time.time() - epoch_timer))
		print("End training, time passed {:.2f}s".format(time.time() - timer))
	elif(args.mode == "infer"):
		# read data and batch them together
		print("Create data for inference...")
		src_file = os.path.join(args.run_dir, args.file_name + "." + args.src)
		batched_dataset, indices = processData(args, src_word_to_id, tgt_word_to_id, src_file, tgt_file=None)
		all_predictions, all_predictions_length = [], []
		# run infer on batched data
		timer = time.time()
		print("Start inference process, with dataset size {:d} of {:d} sentences each".format(len(batched_dataset), args.batch_size))
		for batch in batched_dataset:
			batch_predictions, batch_predictions_length = inferSession(session_data, batch)
			all_predictions.extend(batch_predictions)
			all_predictions_length.extend(batch_predictions_length)
		print("Inference finished, time passed {:.2f}s".format(time.time() - timer))
#		print("First sample: {} - {}".format(all_predictions[0], all_predictions_length[0]))
		# reorder and trim sentences
		timer = time.time()
		tgt_id_to_word = {v:k for k, v in tgt_word_to_id.items() }
		word_lookup_fn = lambda idx: tgt_id_to_word.get(idx, unk_token)
		zip_length_predictions = zip(all_predictions, all_predictions_length)
		# trim
		# because of beam format, it is [batch_size, beam_width, tgt_len] for sentences and [batch_size, beam_width] for sentences_length
		# format to [batch_size, beam_width, correct_length]
		beam_handler = lambda predictions, prediction_lengths: [ pred[:pred_length] for pred, pred_length in zip(predictions, prediction_lengths) ]
		all_predictions = ( beam_handler(preds, pred_lengths) for preds, pred_lengths in zip_length_predictions )
		# take the first beam
		all_predictions = ( batch[0] for batch in all_predictions)
		# reverse lookup and join
		all_predictions = ( " ".join([word_lookup_fn(word_idx) for word_idx in pred]) for pred in all_predictions )
		# sort to correct order
		sorted_all_predictions = sorted(zip(indices, all_predictions))
		_, all_predictions = zip(*sorted_all_predictions)
		with io.open("infer_output.out", "w", encoding="utf8") as infer_file:
			infer_file.write("\n".join(all_predictions))
		print("Reverse lookup and ordered, time passed {:.2f}s".format(time.time() - timer))
	else:
		raise NotImplementedError("Mode {} not coded!".format(args.mode))

