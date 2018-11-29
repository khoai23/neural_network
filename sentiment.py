import tensorflow as tf
import numpy as np
import sys, io, os, time
import json, argparse
# additional functions
from random import shuffle
# the list of possible models
import sentiment_models as model_lib
# hack| load the generate_transform_dict from the import over there
from sentiment_data import loadAndProcessDataset, generate_tranform_dict, custom_vi_cleaner, batch
from ffBuilder import loadFromPath, saveToPath

def readPretrainedEmbedding(file_path):
	with io.open(file_path, "r", encoding="utf8") as emb_file:
		lines = emb_file.readlines()
		embedding_array = []
		word_list = []
		# comprehend the file
		for idx, line in enumerate(lines):
			# replace tab
			line = line.replace("\t"," ")
			if(idx == 0 and len(line.strip().split(" ")) == 2):
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
	with io.open(file_path, "r", encoding="utf8") as json_file:
		json_data = list(json.load(json_file))
	# prepare the cleaner as well
	detector_set, transform_dict = generate_tranform_dict()
	full_data_list = []
	for block in json_data:
		# data is tuple of (input, correct_output)
		# assert input_field in block and output_field in block, "Block error: {}".format(block)
		inputs_batch = block[input_field]
		output_batch = block[output_field]
		# process the input batch using the custom cleaner
		inputs_batch = [custom_vi_cleaner(line, detector_set, transform_dict) for line in inputs_batch]
		full_data_list.extend(zip(inputs_batch, output_batch))
	return full_data_list

def constructParser():
	parser = argparse.ArgumentParser(description='A rewrite of sentiment analysis.')
	parser.add_argument('-m','--mode', type=str, default='default', choices=["default", "data_process", "train", "infer", "eval", "export"], help='Mode of the parser. Default all (default)')
	parser.add_argument('--data_processing_mode', type=str, default='duplicate', choices=["filter", "duplicate", "reliability"], help='Data processing mode. default duplicate')
	parser.add_argument('--data_dir', type=str, default="./", help='Location of files to train, default ./')
	parser.add_argument('-s', '--save_file', type=str, default="default_save", help="Location of file to load/save checkpoints")
	parser.add_argument('--gpu_disable_allow_growth', action="store_true", help='Use to disable gpu_allow_growth')
	parser.add_argument('--debug', action="store_true", help='Use to enable debug information')
	return parser.parse_args()

def buildSession(args):
	# embedding path
	words, embeddings = readPretrainedEmbedding(args.embedding_file)
	default_word = args.default_word
#	print(words)
	default_idx = words.index(default_word)
	print("Default idx detected: {:d}".format(default_idx))
	# initiate entrance point and string_to_id func
	if(args.structure == "attention"):
		return model_lib.buildSessionBidirectionalAttention(words, embeddings, default_idx, cell_size=args.cell_size, debug=args.debug, gpu_allow_growth=not args.gpu_disable_allow_growth)

def exportSession(session, export_path, input_set, output_set, export_sub_folder=0, override=False):
	# create builder
	model_export_path = os.path.join(export_path, str(export_sub_folder))
	if(override and os.path.isdir(model_export_path)):
		print("Exported model detected, override on. Removing..")
		os.rmdir(model_export_path)
	export_builder = tf.saved_model.builder.SavedModelBuilder(model_export_path)
	# load info
	input_set = {k:tf.saved_model.utils.build_tensor_info(v) for k, v in input_set.items()}
	output_set = {k:tf.saved_model.utils.build_tensor_info(v) for k, v in output_set.items()}
	# create signature to be recalled in serving model
	prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=input_set, outputs=output_set, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
	export_builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], signature_def_map={
		"predict_score": prediction_signature
	}, main_op=tf.tables_initializer(), strip_default_attrs=True)
	export_builder.save()
	print("Serve-able model exported at location {}, subfolder {}".format(export_path, export_sub_folder))

if __name__ == "__main__":
	args = constructParser()
	args.data_dir = "/home/quan/Workspace/Data/espresso"
	args.model_dir = "/home/quan/Workspace/Data/model/espresso/default"
	args.embedding_file = "vi_normalized_default.emb.txt"
	args.embedding_file = os.path.join(args.data_dir, args.embedding_file)
	args.data_file = "ttn_data.json_2018-11-12"
	args.data_file = os.path.join(args.data_dir, args.data_file)
#	args.save_file = "test.sav"
	args.save_file = os.path.join(args.model_dir, args.save_file)
	args.export_dir = os.path.join(args.model_dir, "export")
	args.cell_size = 512
	args.structure = "attention"
	args.filter_type = "wer"
	args.epoch = 100
	args.batch_size = 64
	args.dropout = 0.8
	args.filter_range = 1000
	args.filter_threshold = 1.0
	args.shuffle_batch = True
	args.default_word = "<unk>"
	args.force_manual = False
	# add a mode check function to decide which part of the program to be used
	args.check_mode = lambda check: args.mode == check or args.mode == "default"
	args.is_extended_output = False
	score_fn = lambda l1, l2: score(args.filter_type, l1, l2)
	
	block_dump_file = os.path.join(args.model_dir, "block_filtered_data.json")
	data_dump_file = os.path.join(args.model_dir, "fully_filtered_data.json")
	if(args.check_mode("data_process")):
		if(args.mode == "data_process"):
			print("Is data_process mode, force redo all processing of data")
			args.force_manual = True
		print("Load or process data..")
		full_data_load_function = lambda: readJSONData(args.data_file, input_field="ttn_bl_noi_dung", output_field="ttn_bl_diem")
		duplicate_mode = args.data_processing_mode == "duplicate"
		reliability_mode = args.data_processing_mode == "reliability"
		dataset = loadAndProcessDataset(full_data_load_function, data_dump_file, unprocessed_dataset_location=block_dump_file, duplicate_mode=duplicate_mode, reliability_mode=reliability_mode, split_train_and_eval=True, debug=args.debug, force_manual=args.force_manual, extended_output=args.is_extended_output)
		if(isinstance(dataset, dict)):
			print("Data loading process completed (splitted), train {}, eval {}".format(len(dataset["train"]), len(dataset["eval"])))
		else:
			print("Data loading process completed (not splitted), loaded {} lines of comments".format(len(dataset)))
		if(args.mode == "data_process"):
			sys.exit(0)

	session_dictionary = buildSession(args)
	session = session_dictionary["session"]
	input_pl, result_pl, dropout_pl = session_dictionary["placeholders"]
	loss_t, train_op = trainers = session_dictionary["training_ops"] 
	predictions = session_dictionary["predictions"]
	print("Build session done.")
	loadFromPath(session, args.save_file)
	
	# shuffle and create eval/train set
	if(args.check_mode("train")):
		timer = time.time()
		if(isinstance(dataset, dict) and "train" in dataset):
			assert args.is_extended_output == dataset["data_is_extended"], "Data and is_extended_output mismatch: {} vs {}".format(args.is_extended_output, dataset["data_is_extended"])
			print("Train set detected, is_extended_output {}".format(args.is_extended_output))
			train_set_unordered = dataset["train"]
		elif(isinstance(dataset, list)):
			print("Whole set detected, train anyway")
			train_set_unordered = dataset
		# order the set by sentence length 
		train_set_ordered = sorted( train_set_unordered, key=lambda item: len(item[0].split()) )
		training_dropout = args.dropout
		timer = time.time()
		for i in range(args.epoch):
			# start training
			batch_iters = batch(train_set_ordered, n=args.batch_size, shuffle_batch=args.shuffle_batch)
			for batch_idx, iter_batch in enumerate(batch_iters):
				inputs, correct_outputs = zip(*iter_batch)
				inputs = list(inputs)
				if(args.is_extended_output):
					assert np.shape(correct_outputs)[1] == 2, "In extended output mode, the input must be a tuple of score-certainty"
				else:
					correct_outputs = [int(o) for o in correct_outputs]
				loss, _ = session.run(trainers, feed_dict={input_pl:inputs, result_pl:correct_outputs, dropout_pl:training_dropout})
				print("Iter {:d}, batch {:d}, loss {:.4f}, time passed {:.2f}".format(i, batch_idx, loss, time.time() - timer))
			# start eval
			eval_inputs, eval_results = zip(*dataset["eval"])
			eval_predictions = session.run(predictions, feed_dict={input_pl:list(eval_inputs)})
			eval_convert_predictions = [int(pred * 4.0 + 1.5) for pred in eval_predictions]
			eval_results = [int(res) for res in eval_results]
			bundled_eval_data = list(zip(eval_results, eval_convert_predictions))
			eval_set_size = len(bundled_eval_data)
			correct_score = sum( (1 if pred == res else 0 for res, pred in bundled_eval_data) )
			correct_score_string = "{:d}/{:d}({:2.2f}%)".format(correct_score, eval_set_size, float(correct_score) / float(eval_set_size) * 100.0)
			correct_positivity = sum( (1 if (pred == res == 3 or (pred - 3)*(res - 3) > 0 ) else 0 for res, pred in bundled_eval_data) )
			correct_positivity_string = "{:d}/{:d}({:2.2f}%)".format(correct_positivity, eval_set_size, float(correct_positivity) / float(eval_set_size) * 100.0)
			mean_difference = sum( (abs(pred - res) for res, pred in bundled_eval_data) )
			mean_difference_string = "{:d} on {:d} sample (average {:.2f})".format(mean_difference, eval_set_size, float(mean_difference) / float(eval_set_size))
			print("Short Evaluation: Correct score: {:s}, correct positivity {:s}, mean difference {:s}".format(correct_score_string, correct_positivity_string, mean_difference_string))
			# TODO actually sensible global_step var
			saveToPath(session, args.save_file, global_step=(i+1)*batch_idx)
	
	if(args.check_mode("eval")):
		assert isinstance(dataset, dict) and "eval" in dataset, "Must have `eval` property in the dataset"
		print("Begin full evaluation")
		eval_inputs, eval_results = zip(*dataset["eval"])
		eval_predictions = session.run(predictions, feed_dict={input_pl:list(eval_inputs)})
		eval_convert_predictions = (int(pred * 4.0 + 1.5) for pred in eval_predictions)
		eval_results = [int(res) for res in eval_results]
		eval_set_size = len(eval_results)
		eval_full_result = zip(eval_convert_predictions, eval_predictions, eval_results, eval_inputs)
		correct_score = correct_positivity = mean_difference = 0
		print("Full Evaluation:")
		for item in eval_full_result:
			prediction, raw_prediction, result, input_string = item
			if(prediction == result):
				correct_score += 1
				correct_positivity += 1
			else:
				mean_difference += abs(prediction - result)
				if( (prediction - 3) * (result -3) > 0 ):
					correct_positivity += 1
			print("\tpredict {:d}({:.4f}), correct {:d} ||| {:s}".format(*item))
		correct_score_string = "{:d}/{:d}({:2.2f}%)".format(correct_score, eval_set_size, float(correct_score) / float(eval_set_size) * 100.0)
		correct_positivity_string = "{:d}/{:d}({:2.2f}%)".format(correct_positivity, eval_set_size, float(correct_positivity) / float(eval_set_size) * 100.0)
		mean_difference_string = "{:d} on {:d} sample (average {:.2f})".format(mean_difference, eval_set_size, float(mean_difference) / float(eval_set_size))
		print("Total Evaluation: Correct score: {:s}, correct positivity {:s}, mean difference {:s}".format(correct_score_string, correct_positivity_string, mean_difference_string))
	
	if(args.check_mode("export")):
		if(args.is_extended_output):
			raise NotImplementedError()
		else:
			exportSession(session, args.export_dir, {"input": input_pl}, {"output": predictions}, export_sub_folder=0)
