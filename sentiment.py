import tensorflow as tf
import numpy as np
import sys, io, os, re
import json, argparse
# additional functions
# the list of possible models
import sentiment_models as model_lib
# hack| load the generate_transform_dict from the import over there
from sentiment_data import loadAndProcessDataset, generate_tranform_dict, custom_vi_cleaner, NumpySupportedEncoder
from sentiment_dictionary import tryApplyDictionaryToPredictions, loadDictionary
# constant for the moment
MODEL_DEFAULT_LOCATION = "/home/quan/Workspace/Data/model/espresso/default"
EMB_DEFAULT_LOCATION = "/home/quan/Workspace/Data/espresso/vi_trimmed_embedding.txt"
MONOLINGUAL_FILE_LOCATION = "/home/quan/Workspace/Data/monolingual/vi_mono.txt"

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

def readUnknownWords(recognized_words, list_sentences, occurence_floor=5):
	unk_words = {}
	counter_word_nums = 0
	counter_orig_dict = 0
	for sentence in list_sentences:
		for word in sentence.strip().split():
			if word not in recognized_words:
				unk_words[word] = unk_words.get(word, 0) + 1
			else:
				counter_orig_dict += 1
			counter_word_nums += 1
	# filter occurence
	# sort to make sure that the words are in correct order each run, assuming using the same dataset
	additional_words = list(sorted( [word for word, value in unk_words.items() if value >= occurence_floor] ))
	set_additional_words = set(additional_words)
	counter_curr_dict = sum( (1 for sentence in list_sentences for word in sentence.strip().split() if word in recognized_words or word in set_additional_words) )
	print( "Caught {:d} additional words, raising word coverage from {:.2f}% to {:.2f}%".format(len(additional_words), float(counter_orig_dict) / float(counter_word_nums) * 100.0, float(counter_curr_dict) / float(counter_word_nums) * 100) )
	del set_additional_words
	return additional_words

def readJSONData(file_path, input_field="input", output_field="output", duplicate_reduction=False):
	assert os.path.isfile(file_path), "File {:s} not valid!".format(file_path)
	with io.open(file_path, "r", encoding="utf8") as json_file:
		json_data = list(json.load(json_file))
	# prepare the cleaner as well
	detector_set, transform_dict = generate_tranform_dict()
	if(duplicate_reduction):
		print("Using duplicate_reduction in readJSONData")
		# turn list of dots into long_dot
		long_dot_regex = re.compile(r"\.{2,}")
		# reduce all words to its own self
		duplicate_word_regex = re.compile(r"(\D)\1+")
#		duplicate_symbol_regex = re.compile("([\(\)\?\!,_\-\"])\1+")
		# replace all the !/?/, accompanied by .
		redundant_regex = re.compile(r"([\?\!,])\.")
		el_symbol_regex = re.compile("\u0001 *")
		# sao regex
		star_regex = re.compile(r"(\d)(sao|\*|\u2605\u2606)")
		def cleaner(line):
#			orig_line = line
			line = re.sub(long_dot_regex, " … ", line)
#			search_result = re.search(duplicate_word_regex, line)
			line = re.sub(duplicate_word_regex, r"\1", line)
#			line = re.sub(duplicate_symbol_regex, "\1", line)
			line = re.sub(redundant_regex, r"\1", line)
			line = re.sub(star_regex, r"\1 sao", line)
			line = custom_vi_cleaner(line, detector_set, transform_dict)
			line = re.sub(el_symbol_regex, "", line)
#			print("orig_line > line: {:s} > {:s}".format(orig_line, line))
			return line
	else:
		cleaner = lambda line: custom_vi_cleaner(line, detector_set, transform_dict)
	full_data_list = []
	for block in json_data:
		# data is tuple of (input, correct_output)
		# assert input_field in block and output_field in block, "Block error: {}".format(block)
		inputs_batch = block[input_field]
		output_batch = block[output_field]
		# process the input batch using the custom cleaner
		inputs_batch = [cleaner(line) for line in inputs_batch]
		full_data_list.extend(zip(inputs_batch, output_batch))
	return full_data_list

def readRawData(folder, train_folder="train", test_folder="test", train_file_format="training_{:s}.txt", test_file="test.txt"):
	assert os.path.isdir(folder), "Path is not folder"
	# prepare cleaner
	detector_set, transform_dict = generate_tranform_dict()
	train_data_list = []
	for file_type, rating_type in zip(["positive", "neutral", "negative"], ["5", "3", "1"]):
		with io.open(os.path.join(folder, train_folder, train_file_format.format(file_type)), "r", encoding="utf-8") as read_file:
			# the blank are removed in sentiment_data process, but just in case
			train_data_list.extend( ((custom_vi_cleaner(line, detector_set, transform_dict), rating_type) for line in read_file.readlines() if line.strip() != "") )
	eval_data_list = []
	if("{:s}" in test_file):
		print("Test file is formatable, do as train")
		for file_type, rating_type in zip(["positive", "neutral", "negative"], ["5", "3", "1"]):
			with io.open(os.path.join(folder, train_folder, train_file_format.format(file_type)), "r", encoding="utf-8") as read_file:
				# the blank are removed in sentiment_data process, but just in case
				eval_data_list.extend( ((custom_vi_cleaner(line, detector_set, transform_dict), rating_type) for line in read_file.readlines() if line.strip() != "") )
	else:
		print("Test file is non-formatable, read as one big test")
		with io.open(os.path.join(folder, test_folder, test_file), "r", encoding="utf-8") as read_file:
			score_convert_dict = {"POS":"5", "NEG":"1", "NEU":"3"}
			for test_line, answer_line in zip(read_file, read_file):
				score = score_convert_dict[answer_line.strip()]
				# read two lines at once, and add them into the eval set
				eval_data_list.append( (custom_vi_cleaner(test_line, detector_set, transform_dict), score) )
#	for line, score in eval_data_list:
#		print("{} ({})".format(line, score))
	return train_data_list, eval_data_list

ALL_LOCATIONS = [
	{"type":"folder", "name":"SA-2016", "path":"/home/quan/Workspace/Data/espresso", "train_folder":"SA2016-training_data", "test_folder":"SA2016-TestData-Ans/SA2016-TestData-Ans", "train_file_format":"SA-training_{:s}.txt", "test_file":"test_raw_ANS.txt"},
	{"type":"folder", "name":"IMDB", "path":"/home/quan/Workspace/Data/espresso/aclImdb", "train_folder":"train", "test_folder":"test", "train_file_format":"{:s}.txt", "test_file":"{:s}.txt"},
	{"type":"file", "name":"json_test", "path":"/home/quan/Workspace/Data/espresso/ttn_data.json_2018-11-12", "input_field":"ttn_bl_noi_dung", "output_field":"ttn_bl_diem"},
	{"type":"file", "name":"json_10k", "path":"/home/quan/Workspace/Data/espresso/tsv_manually_annotated/json_10k.txt", "input_field":"lines", "output_field":"ratings", "duplicate_reduction":True},
	{"type":"file", "name":"local_json_10k", "path":"./data/json_10k.txt", "input_field":"lines", "output_field":"ratings", "duplicate_reduction":True},
	{"type":"files", "name":"json_20k", "path":["/home/quan/Workspace/Data/espresso/tsv_manually_annotated/json_10k.txt", "/home/quan/Workspace/Data/espresso/tsv_manually_annotated/json_10k_problem.txt"], "input_field":"lines", "output_field":"ratings", "duplicate_reduction":True},
	{"type":"files", "name":"local_json_20k", "path":["./data/json_10k.txt", "./data/json_10k_problem.txt"], "input_field":"lines", "output_field":"ratings", "duplicate_reduction":True},
	{"type":"file", "name":"json_combined", "path":"/home/quan/Workspace/Data/espresso/tsv_manually_annotated/json_joined.txt", "input_field":"lines", "output_field":"ratings", "duplicate_reduction":True},
]

def tryReadDataFromPath(data_config):
	if(isinstance(data_config, str)):
		data_config = next((conf for conf in ALL_LOCATIONS if conf["name"] == data_config))
	# currently no customization option.
	path_type = data_config.pop("type")
	data_config.pop("name")
	path = data_config.pop("path")
	if(isinstance(path, (list, tuple, set)) and path_type=="files"):
		print("Path list {} is multitude of files, use JSON reading format".format(path))
		data = []
		for p in path:
			data.extend( readJSONData(p, **data_config) )
		return data
	elif(os.path.isdir(path) and path_type == "folder"):
		print("Path {:s} is directory, use SA-2016 format".format(path))
		return readRawData(path, **data_config)
	elif(os.path.isfile(path) and path_type == "file"):
		print("Path {:s} is single file, use JSON reading format".format(path))
		return readJSONData(path, **data_config)
	else:
		raise ValueError("Path {:s} invalid, mismatch with {}".format(path, path_type))

def constructParser():
	parser = argparse.ArgumentParser(description='A rewrite of sentiment analysis.')
	parser.add_argument('-m','--mode', type=str, default='default', choices=["default", "data_process", "train", "infer", "tune", "eval", "export", "debug"], help='Mode of the parser. Default all (default)')
	parser.add_argument('--data_processing_mode', type=str, default='reliability', choices=["filter", "duplicate", "reliability"], help='Data processing mode. default reliability')
	parser.add_argument('--model_structure', type=str, default='attention', choices=["attention", "attention_extended", "attention_extended_v2", "attention_multimodal", "sum"], help='The model structure to be used in sentiment model. Default attention')
#	parser.add_argument('--data_file', type=str, default=DATA_DEFAULT_LOCATION, help='Location of files to train, default ./')
	parser.add_argument('--data_config', type=str, choices=[conf["name"] for conf in ALL_LOCATIONS], default=None, help='Location of files to train, default ./')
	parser.add_argument('--model_dir', type=str, default="./", help='Location of files to train, default ./')
	parser.add_argument('--gpu_disable_allow_growth', action="store_true", help='Use to disable gpu_allow_growth')
	parser.add_argument('--debug', action="store_true", help='Use to enable debug information')
	parser.add_argument('--force_manual', action="store_true", help='Enable to force rebuilding dataset')
	parser.add_argument('--use_trained_embedding', action="store_true", help='Enable to use trained embeddings, specified with trained_embedding_location')
	parser.add_argument('--trained_embedding_location', type=str, default=EMB_DEFAULT_LOCATION, help='The location of the embedding to be used.')
	parser.add_argument('--checkpoint_index', type=str, default=None, help='The index of the checkpoint to be retrieved for eval/export. Default None (latest)')
	parser.add_argument('--balance_reduce_mode', type=str, choices=["2worst", "partial", "avg", "none"], default="2worst", help='Only in data_processing_mode reliability. The method used to balance the dataset. See sentiment_data\'s balanceSetByReduction for detail')
	parser.add_argument('--use_weighted_loss', action="store_true", help='Enable to use weighted loss, currently only available to attention_extended')
	parser.add_argument('--use_monolingual_data', action="store_true", help='Enable to use external monolingual data as neutral(3)')
	parser.add_argument('--monolingual_data_location', type=str, default=MONOLINGUAL_FILE_LOCATION, help='Location for the monolingual data')
	parser.add_argument('--use_tuning', action="store_true", help='Use tuned parameters to export the rating value')
	parser.add_argument('--tuning_correction', action="store_true", help='Use correction value instead of 1-1')
	# extra arguments dependent on modes
	parser.add_argument('--eval_print_alignment', action="store_true", help='Enable to eval by saving an alignment image to a specified location')
	parser.add_argument('--external_dictionary', type=str, default=None, help='If specified, use an external dictionary to rectify the result.')
	parser.add_argument('--export_override', action="store_true", help='If true, override the export folder as needed')
	return parser.parse_args()

def createModel(args, dataset):
	# embedding path
	if(args.use_trained_embedding):
		words, embeddings = readPretrainedEmbedding(args.trained_embedding_location)
		# pretrained, value is the index (int)
		default_word_idx = words.index(args.default_word)
		print("Default idx detected: {:d}".format(default_word_idx))
	else:
		words, embeddings = [], args.cell_size
		# untrained, value is the word (str)
		default_word_idx = args.default_word
		print("Default word set: {:s}".format(default_word_idx))
	# read the dataset sentences and create the new words base on it
	if(isinstance(dataset, dict) and "train" in dataset and "eval" in dataset):
		print("Dataset is splitted dict")
		dataset_sentences, dataset_scores, *_ = zip( *(dataset["train"] + dataset["eval"]) )
	elif(isinstance(dataset, list)):
		print("Dataset is unsplitted list")
		dataset_sentences, dataset_scores = zip( *dataset )
	else:
		raise ValueError("Dataset error, please recheck")
	# initiate entrance point and string_to_id func
	additional_words = readUnknownWords(set(words), dataset_sentences)
	# use the dataset_score to create the weighted_loss dict
	if(args.use_weighted_loss):
		weighted_loss = {}
		for score in dataset_scores:
			weighted_loss[score] = weighted_loss.get(score, 0) + 1
	else:
		weighted_loss = None

	if(args.model_structure == "attention"):
		model = model_lib.SentimentRNNAttention(args.save_file, args.export_dir, batch_size=args.batch_size, debug=args.debug, shuffle_batch=args.shuffle_batch, loss_weight=weighted_loss)
		model.buildSession(words, embeddings, default_word_idx, cell_size=args.cell_size, additional_words=additional_words, gpu_allow_growth=not args.gpu_disable_allow_growth)
	elif(args.model_structure == "attention_extended"):
		args.use_certainty = False
		model = model_lib.SentimentRNNAttentionExtended(args.save_file, args.export_dir, batch_size=args.batch_size, debug=args.debug, shuffle_batch=args.shuffle_batch, loss_weight=weighted_loss, use_certainty_loss=args.use_certainty)
		model.buildSession(words, embeddings, default_word_idx, cell_size=args.cell_size, additional_words=additional_words, gpu_allow_growth=not args.gpu_disable_allow_growth)
	elif(args.model_structure == "attention_extended_v2"):
		args.use_certainty = False
		model = model_lib.SentimentRNNAttentionExtendedV2(args.save_file, args.export_dir, batch_size=args.batch_size, debug=args.debug, shuffle_batch=args.shuffle_batch, loss_weight=weighted_loss, use_certainty_loss=args.use_certainty)
		model.buildSession(words, embeddings, default_word_idx, cell_size=args.cell_size, additional_words=additional_words, gpu_allow_growth=not args.gpu_disable_allow_growth)
	elif(args.model_structure == "attention_multimodal"):
		model = model_lib.SentimentRNNAttentionMultimodal(args.save_file, args.export_dir, batch_size=args.batch_size, debug=args.debug, shuffle_batch=args.shuffle_batch)
		model.buildSession(words, embeddings, default_word_idx, cell_size=args.cell_size, additional_words=additional_words, gpu_allow_growth=not args.gpu_disable_allow_growth)
	elif(args.model_structure == "sum"):
		model = model_lib.SentimentRNNBalanceModel(args.save_file, args.export_dir, batch_size=args.batch_size, debug=args.debug, shuffle_batch=args.shuffle_batch)
		model.buildSession(words, embeddings, default_word_idx, cell_size=args.cell_size, additional_words=additional_words, gpu_allow_growth=not args.gpu_disable_allow_growth)
	else:
		raise NotImplementedError("Model structure not recognized: {:s}".format(args.model_structure))
	return model

if __name__ == "__main__":
	args = constructParser()
	if(args.model_dir == None):
		args.model_dir = MODEL_DEFAULT_LOCATION
		print("No model directory specified, reset to default location: {}".format(args.model_dir))
	if(not os.path.isdir(args.model_dir)):
		print("Model folder not ready, creating new...")
		os.makedirs(args.model_dir)
#	args.embedding_file = os.path.join(args.data_dir, args.embedding_file)
	args.save_file = os.path.join(args.model_dir, "default_save")
	args.export_dir = os.path.join(args.model_dir, "export")
	args.cell_size = 128
	args.epoch = 100
	args.batch_size = 64
	args.dropout = 0.8
	args.shuffle_batch = True
	args.default_word = "<unk>"
	# add a mode check function to decide which part of the program to be used
	args.check_mode = lambda check: args.mode == check or args.mode == "default"
	if( any(( args.model_structure == item for item in ["attention_extended", "attention_multimodal", "attention_extended_v2"])) ):
		print("Model must use extended output. Setting it to True")
		args.is_extended_output = True
	else:
		args.is_extended_output = False
	
	block_dump_file = os.path.join(args.model_dir, "block_filtered_data.json")
	data_dump_file = os.path.join(args.model_dir, "fully_filtered_data.json")
	if( True or args.check_mode("data_process")):
		if(args.mode == "data_process"):
			print("Is data_process mode, force redo all processing of data")
			args.force_manual = True
		print("Load or process data..")
		duplicate_mode = args.data_processing_mode == "duplicate"
		reliability_mode = args.data_processing_mode == "reliability"
		full_data_load_function = lambda: tryReadDataFromPath(args.data_config)
		dataset = loadAndProcessDataset(full_data_load_function, data_dump_file, unprocessed_dataset_location=block_dump_file, duplicate_mode=duplicate_mode, reliability_mode=reliability_mode, split_train_and_eval=True, debug=args.debug, force_manual=args.force_manual, extended_output=args.is_extended_output, reduce_mode=args.balance_reduce_mode, monolingual_file=args.monolingual_data_location if args.use_monolingual_data else None)
		if(isinstance(dataset, dict)):
			print("Data loading process completed (splitted), train {}, eval {}".format(len(dataset["train"]), len(dataset["eval"])))
		else:
			print("Data loading process completed (not splitted), loaded {} lines of comments".format(len(dataset)))
		if(args.mode == "data_process"):
			print("Data process mode, done. Exiting")
			sys.exit(0)

	sentiment_model = createModel(args, dataset)
	session = sentiment_model._session_dictionary["session"]
	session.run([tf.global_variables_initializer(), tf.tables_initializer()])
	sentiment_model.loadSession()

	print("Build and attempt load session done.")
	if(args.mode == "debug"):
#		with io.open("graph_str.txt", "w", encoding="utf-8") as graph_file:
#			model_lib.exportMetaGraph(graph_file)
		session_values_file = "_session_value_file_temp.json"
		temp_file = "_temp.json"
		value_dict = model_lib.trySaveSessionValues(session)
#		all_variables = tf.trainable_variables()
#		all_var_values = session.run(all_variables)
#		value_dict = {var.name:val for var, val in zip(all_variables, all_var_values)}
		if(os.path.isfile(session_values_file)):
			# dump for conversion
			temp_file = io.StringIO()
			json.dump(value_dict, temp_file, cls=NumpySupportedEncoder)
			del value_dict
			temp_file.seek(0)
			value_dict = json.load(temp_file)
			del temp_file
			# load from past file
			with io.open(session_values_file, "r", encoding="utf-8") as past_value_file:
				past_value_dict = json.load(past_value_file)
#			os.remove(session_values_file)
			# compare
			key_set = set().union(set(value_dict.keys()), set(past_value_dict.keys()))
			for key in sorted(key_set):
				current_val = np.asarray( value_dict.get(key, None) )
				past_val = np.asarray( past_value_dict.get(key, None) )
				if(np.array_equal(current_val, past_val)):
					print("( ) Key {} consistent.".format(key))
#					pass
				else:
					print("(X) Key {} inconsistent".format(key))
					print("Key comp: {}".format( np.equal(current_val.flatten(), past_val.flatten()) ))
		else:
			with io.open(session_values_file, "w", encoding="utf-8") as value_file:
				json.dump(value_dict, value_file, cls=NumpySupportedEncoder)
			print("Dumped current run to {:s}".format(session_values_file))
		sys.exit(0)
	
	# diff and locate the train/eval set
	if(isinstance(dataset, dict) and "train" in dataset):
		assert args.is_extended_output == dataset["data_is_extended"], "Data and is_extended_output mismatch, recommend a rerun using force_manual: {} vs {}".format(args.is_extended_output, dataset["data_is_extended"])
		print("Train set detected, is_extended_output {}".format(args.is_extended_output))
		train_set_unordered = dataset["train"]
		eval_set = dataset["eval"]
	elif(isinstance(dataset, list)):
		print("Whole set detected, train anyway")
		train_set_unordered = dataset
		eval_set = None
	# shuffle and create eval/train set
	if(args.check_mode("train")):
		# order the set by sentence length 
		train_set_ordered = sorted( train_set_unordered, key=lambda item: len(item[0].split()) )
		sentiment_model.trainSession(train_set_ordered, eval_set, args.epoch, dropout=args.dropout)
	
	if(args.mode != "default"):
		# in eval/export portion, must load checkpoints from saved location if specified
		if(args.checkpoint_index):
			print("Loading specific checkpoint: {:s} @{:s}".format(args.checkpoint_index, args.save_file))
			sentiment_model.loadSession(args.checkpoint_index)

	if((args.check_mode("tune") or args.use_tuning) and eval_set):
		print("Run tuning on eval set")
		sentiment_model.tune(eval_set, weight_correction=args.tuning_correction, debug=args.debug)

	if(args.check_mode("eval") and eval_set):
		if(args.external_dictionary):
			# force to load only, will throw exception if false
			dictionary = loadDictionary(args.external_dictionary, force_reload=False, dict_paths=None, rev_paths=None)
			rectifier_fn = lambda sents, rats: tryApplyDictionaryToPredictions(sents, rats, dictionary, debug=args.debug)
		else:
			rectifier_fn = None
		sentiment_model.evalSession(eval_set, detailed=True, alignment_sample=args.eval_print_alignment, alignment_file_path=args.save_file, external_fn=rectifier_fn)
	
	if(args.check_mode("export")):
		checkpoint = args.checkpoint_index if args.checkpoint_index else 0
		sentiment_model.exportSession(export_sub_folder=checkpoint, override=args.export_override)
