# cleaner functions
from scripts.cleaner import generate_tranform_dict, vietnamese_ghost_characters_cleaner, split_emoticon, remove_html_tags, misc_reformat
from scripts.tokenize_word import clean
from xer import levenshtein
from random import shuffle
import os, io, json, itertools
import numpy as np
from sentiment_eval import createDistributionGraph

generate_tranform_dict = generate_tranform_dict
#from json import JSONEncoder
class NumpySupportedEncoder(json.JSONEncoder):
	"""Because fuck dumping numpy object."""
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, (bytes, bytearray)):
			return str(obj)
		elif isinstance(obj, (np.bool, np.bool_)):
			return bool(obj)
		else:
			print(type(obj))
			return super(NumpySupportedEncoder, self).default(obj)

def custom_vi_cleaner(line, detector_set, transform_dict):
#	orig_line = line
	# lowercase, remove diacritics and separate punctuation
	line = remove_html_tags(line)
	line = vietnamese_ghost_characters_cleaner(line, detector_set, transform_dict, ignore_error=True)
	line = split_emoticon(line)
	cleaned_line = clean(line).lower()
#	print("Clean: {} -> {}".format(orig_line, cleaned_line))
	return cleaned_line
def score(filter_type, l1, l2):
	assert filter_type == "wer" or filter_type == "cer"
	# Compare l1, l2 by filter type
	if(filter_type == "wer"):
		l1, l2 = l1.split(), l2.split()
	_, (s, i, d) = levenshtein(l1, l2)
	return 1.0 - ( (s + i + d) / max(len(l1), len(l2), 1.0) )

def statDataset(dataset):
	# if the score value is single string/int
	data_stat = {}
	for item in dataset:
		score = item[1]
		if(isinstance(score, list) and not isinstance(score, str)):
			for sc in score:
				data_stat[sc] = data_stat.get(score, 0) + 1
		else:
			data_stat[score] = data_stat.get(score, 0) + 1
	return data_stat

def statDuplicateDataset(dataset):
	# if the scores value is set
	data_stat_correct = {}
	data_stat_possible = {}
	for item in dataset:
		scores = item[1]
		data_dict = data_stat_correct if len(scores) == 1 else data_stat_possible
		for score in scores:
			data_dict[score] = data_dict.get(score, 0) + 1
	return data_stat_correct, data_stat_possible

def statReliabilityDataset(dataset):
	# if the scores value is dict
	total_stat = {}
	for item in dataset:
		scores = item[1]
		for key in scores:
			total_stat[key] = total_stat.get(key, 0) + scores[key]
	return total_stat

def balanceSetByReduction(dataset, debug=False, reduce_mode="2worst"):
	stats = statDataset(dataset)
	categorized_dataset = {k:[item for item in dataset if item[1] == k] for k in stats.keys()}
	if(reduce_mode == "avg"):
		# reduce to an approx. maximum of 20% of dataset size
		# assume 50% skew on 5, that will be reduced to 20%, which is around 28% total
		# if it is too high (80%), the dataset will still reduce it to around 50%. Bad move
		reduce_value = len(dataset) // len(categorized_dataset) + 10
	elif(reduce_mode == "2worst"):
		# reduce to the sum of the two worst items
		# might be too aggressive, since even rating 4 is affected by the trim
		reduce_value = sum( list( sorted([val for val in stats.values()]) )[:2] )
	elif(reduce_mode == "partial"):
		# reduce only the highest to 40% of AFTER set
		after_set_prefered_size = sum(list( sorted([val for val in stats.values()]) )[:-1]) * len(stats) / (len(stats) - 1)
		reduce_value = after_set_prefered_size * 4 / 10
	elif(reduce_mode == "none"):
		reduce_value = -1
	else:
		raise ValueError("Reduce mode {} unsupported".format(reduce_mode))
	reduced_dataset = {k:v[:reduce_value] for k, v in categorized_dataset.items()}
	del categorized_dataset
	balanced_dataset = [item for score, items in reduced_dataset.items() for item in items]
	balanced_dataset_stats = statDataset(balanced_dataset)
	if(debug):
		print("Dataset reduced from {} to {}".format(stats, balanced_dataset_stats))
	return balanced_dataset

def _random_select_score_function(score_list, score_weights):
	# use the score weight to select basing on the score_list
	if(len(score_list) == 1):
		return score_list[0]
	else:
		probs = [score_weights[score] for score in score_list]
		sum_probs = sum(probs)
		probs = [p / sum_probs for p in probs]
		return np.random.choice(score_list, 1, p=probs)[0]

def balanceSetByPossibleElement(dataset, select_score_fn=_random_select_score_function, certainty_score=False, debug=False):
	# calculate a sort of scaling percentage to assign the values of possibles
	# we default so that the more a score have in {correct}, the less possibility it is chosen in {possible}
	correct_stats, _ = statDuplicateDataset(dataset)
	correct_deviation = max(min(correct_stats.values()) // 100, 10)
	correct_ceiling = max(correct_stats.values())
	# calculate the score weight and store them in a dict
	score_weights = { k:float(correct_ceiling+correct_deviation-v) for k,v in correct_stats.items() }
	# run through the dataset and select using the selectScore func
	if(certainty_score):
		# add the certainty score basing on the number of item in scores
		# 0.0 when scoreList have all possible score, 1.0 when it have only one
		certainty_scaling = float(len(score_weights) - 1)
		certainty_score_fn = lambda scoreList: 1.0 - float(len(scoreList) - 1) / certainty_scaling
		result_dataset = [ (line, select_score_fn(scores, score_weights), certainty_score_fn(scores)) for line, scores in dataset]
	else:
		result_dataset = [ (line, select_score_fn(scores, score_weights)) for line, scores in dataset]
	if(debug):
		balanced_stats = statDataset(result_dataset)
		print("Data duplicate balance statistic:   correct: {}, balanced by assigning possible elements: {}".format(correct_stats, balanced_stats))
	return result_dataset

def filterData(data, dump_stream_fn=None, keep_duplicate=False, debug=False):
	# assume data is cleaned and tokenized
	# filter only strict duplicate
	# TODO save the filtered data in form of occurence of duplicates (12 case of 1, 4 case of 2..)
	filtered_item_list = []
	# conflict items keep a set of indexes refering to the filtered item list
	if(keep_duplicate):
		conflict_items = dict()
	else:
		conflict_items = set()
	for idx, item in enumerate(data):
		if(item[0] == ""):
			# prevent empty string data
			print("Item {} empty!".format(idx))
			pass
		# check with all passed
		checker = filtered_item_list
		duplicate_found = False
		for check_idx, check_item in enumerate(checker):
			# comparing inputs
			duplicate_found = check_item[0] == item[0]
			if(duplicate_found):
#				print("Full duplicate found @ idx {}-{}".format(idx, check_idx))
				break
		if(not duplicate_found):
			# no duplication, add to filtered items
			filtered_item_list.append(item)
			if(dump_stream_fn):
				# have an external dump stream where it can write the new sentences into
				dump_stream_fn(item, idx)
		elif(item[1] != check_item[1] and (check_idx not in conflict_items or (keep_duplicate and item[1] not in conflict_items[check_idx])) ):
			# go into this block only when mismatch is detected and it is either (1) not keep duplicate and first found, or (2) do keep duplicate and value is not in set
			if(keep_duplicate):
				conflict_item_scores = conflict_items.get(check_idx, set(check_item[1]))
				conflict_item_scores.add(item[1])
				# keep the indexes in the conflict_items dict for later retrieval
				conflict_items[check_idx] = conflict_item_scores
				if(debug):
					print("Rating issue detected: check item {}({}) has rating {} while current item has rating(s) {}".format(idx, item[0], item[1], conflict_item_scores))
			else:
				conflict_items.add(check_idx)
				if(debug):
					print("Mismatch issue detected: check item {}({}) has rating {} while current item has rating {}".format(idx, item[0], item[1], check_item[1]))
	# remove the conflict items, from the highest indices to avoid selecting wrong element ([1, 2, 3, 4] remove item 1, 3 will result in error since you are deleting 3 from [1, 3, 4])
	if(keep_duplicate):
		print("Replace the duplicate with a set of values")
		for conflict_idx, value_set in conflict_items.items():
			filtered_item_list[conflict_idx] = (filtered_item_list[conflict_idx][0], value_set)
	else:
		print("Remove the duplicates altogether")
		for conflict_idx in sorted(conflict_items, reverse=True):
			del filtered_item_list[conflict_idx]
	return filtered_item_list

def reduceData(data, dump_stream_fn=None, debug=False):
	# iterate through data, and keep a dict of scores for each comment
	# like filterData, assume that the lines is already processed
	data_dict = {}
	for idx, (line, score) in enumerate(data):
		if(line not in data_dict):
			if(debug):
				print("Newline {} found at idx {}".format(len(data_dict), idx))
			dump_stream_fn and dump_stream_fn(line)
		line_dict = data_dict.get(line, {})
		line_dict[score] = line_dict.get(score, 0) + 1
		data_dict[line] = line_dict
	return data_dict.items()

def _default_certainty_function(score_dict):
	# default: certainty is the variation below
	return float( max(score_dict.values()) ) / float( sum(score_dict.values()) )

def _default_reliability_function(score_dict):
	# default: base the reliability on the rarity 1 / (sum(score)) and the variation (highest_score/sum(score))
	# it is then highest_score/sum(scores) + 1 / sum(scores)
#	rarity_score = 1.0 / float(sum( score_dict.values() ))
#	variation_score = max( score_dict.values() ) * rarity_score
	return float( max(score_dict.values()) + 1 ) / float( sum(score_dict.values()) * 2)

def _default_select_score_function(score_dict):
	# default: select the case with the highest
	return max(score_dict.items(), key=lambda x: x[1])[0]

def balanceSetByAllScores(data, reliability_fn=_default_reliability_function, select_score_fn=_default_select_score_function, certainty_score=False, reduce_mode="2worst", debug=False):
	if(debug):
		stats = statReliabilityDataset(data)
	# first, for each comment, choose the score basing on the select_score_fn and the reliability rating
	data_values = ( (reliability_fn(score_dict), select_score_fn(score_dict)) for line, score_dict in data )
	# bundle the set
	bundled_data = [ ((line, rel_rating), score) for (rel_rating, score), (line, score_dict) in zip(data_values, data)]
	# sort the data basing on the rating in the descending order
	rating_sorted_data = sorted(bundled_data, key=lambda item: item[0][1], reverse=True)
	# run the reduce as normal, this will clip the data and keep those with the better rating(surer and rarer)
	reduced_dataset = balanceSetByReduction(rating_sorted_data, reduce_mode=reduce_mode)
	# un-bundle the set, currently discarding the rel_rating
	if(certainty_score):
		# keep the rel_rating
		result_dataset = [ (line, score, rel_rating) for (line, rel_rating), score in reduced_dataset]
	else:
		result_dataset = [ (line, score) for (line, rel_rating), score in reduced_dataset]
	if(debug):
		result_stats = statDataset(result_dataset)
		print("Balance attempt: from overall {} to {}".format(stats, result_stats))
	return result_dataset

def _count_vocab_coverage(tokens, vocab):
	if(len(tokens) == 0):
		return 0.0
	return float(len([tok for tok in tokens if tok in vocab])) / float(len(tokens))

def _count_vocab_preference(tokens, vocab):
	if(len(tokens) == 0):
		return 0.0
	return sum((vocab.get(token, 0.0) for token in tokens)) / float(len(tokens))

def _safe_split(line):
	"""Split only if there is space"""
	if(line.find(" ") > -1):
		return line.strip().split()
	else:
		return [line.strip()]

def addMonolingualAsNeutral(old_dataset, monolingual_file, adding_scheme="largest"):
	assert os.path.isfile(monolingual_file), "Monolingual file not found: {:s}".format(monolingual_file)
	# read the old dataset for the vocab and the neutral sentences to be added
	sentences, ratings, *_ = zip(*old_dataset)
	# vocab building
	words = (word for sentence in sentences for word in sentence.strip().split())
	vocab, rat_count = dict(), dict()
	for word in words:
		vocab[word] = vocab.get(word, 0) + 1
	if(adding_scheme == "largest"):
		# count dupl tokens
		vocab = {k for k, v in vocab.items() if v >= 5}
		scorer_fn = lambda s: _count_vocab_coverage(_safe_split(s), vocab)
	elif(adding_scheme == "preference"):
		# count tokens basing on its commonness in the original dataset
		vocab_full_size = float(sum(vocab.values()))
		vocab = {k:float(v)/vocab_full_size for k, v in vocab.items() if v >= 5}
		scorer_fn = lambda s: _count_vocab_preference(_safe_split(s), vocab)
	else:
		raise ValueError("Unknown adding_scheme: {:s}".format(adding_scheme))
	for rat in ratings:
		rat_count[rat] = rat_count.get(rat, 0) + 1
	neutral_needed = max((val for val in rat_count.values())) - rat_count["3"] 
	print("Maximum neutral sentences to be added into the dataset: {:d}".format(neutral_needed))
	if(neutral_needed == 0):
		# fast escape
		return old_dataset
	# rate the sentences in the monolingual file basing on the vocab coverage, and taking the needed ones
	with io.open(monolingual_file, "r", encoding="utf-8") as mono_file:
		best_lines = sorted((misc_reformat(line) for line in mono_file.readlines()), key=scorer_fn)
		# remove those too short (10 tok or < 40 char)
		best_lines = (line for line in best_lines if len(_safe_split(line)) > 10 and len(line) > 40 )
		selected_lines = list(itertools.islice(best_lines, neutral_needed))
		print("Actual neutral sentences found: {:d}".format(len(selected_lines)))
	item_format_size = len(old_dataset[0])
	# append to data
	if(item_format_size == 2):
		print("Filter/duplicate mode detected @ addMonolingualAsNeutral")
		old_dataset.extend(((line, "3") for line in selected_lines))
	elif(item_format_size == 3):
		print("Reliablity mode detected @ addMonolingualAsNeutral")
		old_dataset.extend(((line, "3", 0.1) for line in selected_lines))
	return old_dataset

def loadAndProcessDataset(raw_dataset_call_fn, save_location, duplicate_mode=False, reliability_mode=False, split_train_and_eval=False, debug=False, unprocessed_dataset_location=None, force_manual=False, extended_output=False, save_dataset_pretty=True, reduce_mode="2worst", monolingual_file=None):
	"""The main function to process data
		Args:
			raw_dataset_call_fn: if no file at unprocessed_dataset_location and save_location, call this to receive the raw data
			save_location: location to load if previously run, else dump result to it
			duplicate_mode: if true, line with multiple scores will select the one that balance the dataset
			reliability_mode: if true, line with multiple scores will select the highest, and also favor lines that is (1) unique and (2) certain
			split_train_and_eval: if true, split the dataset to train/eval set
			debug: if true, inner process's debug flag is activate
			unprocessed_dataset_location: if specified and processing raw data, load the raw from here or dump these data in here
			force_manual: if true, override all prior processing (no json load). Still dump to these location after load
			extended_output: if true, the value is 2d numpy of score(int 1-5) and certainty(float 0.0-1.0). Used in extended mode 
			save_dataset_pretty: The dumped file will have pretty print dataset, set to False to disable
			reduce_mode: the method to balance the dataset by reduction. Possible options are 2worst|avg|partial. Default 2worst
			monolingual_file: if specified, load extra sentences in this file as neutral sentences (3)
		Returns:
			If have dump file, load it up regardless of option
			If split, return a dict containing `train` and `eval`
			If not, return the full dict
			"""
	assert not (extended_output and not split_train_and_eval), "extended_output must have split_train_and_eval value as True"
	json_dump_kwargs = {"sort_keys":True, "indent": 4} if save_dataset_pretty else {}
	# Check if save_location have past processed data, if true, load and exit
	if(os.path.isfile(save_location) and not force_manual):
		print("Saved data located @ {}".format(save_location))
		with io.open(save_location, "r", encoding="utf-8") as saved_data_file:
			return json.load(saved_data_file)
	# Check if unprocessed_dataset_location have raw unprocessed data, if yes, supersede the load from raw_dataset_location
	print("Load/Make filtered data")
	if(unprocessed_dataset_location and os.path.isfile(unprocessed_dataset_location) and not force_manual):
		print("Detect filtered data saved @{}".format(unprocessed_dataset_location))
		with io.open(unprocessed_dataset_location, "r", encoding="utf-8") as raw_dataset_file:
			dataset = json.load(raw_dataset_file)
		# this to process if passed block but not filter
		eval_set_size = None
	else:
		print("Make data from raw source...")
		raw_dataset = raw_dataset_call_fn()
		if(len(raw_dataset) == 2 and isinstance(raw_dataset, tuple)):
			print("Train/Eval splitted, process the dataset as a whole and rejoin later")
			train_set, eval_set = raw_dataset
			eval_set_size = len(eval_set)
			dataset = eval_set + train_set
			if(reliability_mode):
				print("Create false certainty value exclusively for reliability mode")
				dataset = [(line, score, 0.0) for line, score in dataset]
		else:
			eval_set_size = None
			if(reliability_mode):
				# process raw data in reduceData (line-score_dict)
				dataset = reduceData(raw_dataset, debug=debug)
			else:
				# process in either keep/filter duplicate
				dataset = filterData(raw_dataset, keep_duplicate=duplicate_mode, debug=debug)
		if(unprocessed_dataset_location):
			print("Dumping filtered data @ {}(overwrite)".format(unprocessed_dataset_location))
			with io.open(unprocessed_dataset_location, "w", encoding="utf-8") as raw_dataset_file:
				# keep_duplicate second var is a set, turn it to a list so as to make it dump-able
				if duplicate_mode:
					dumpable_dataset = [(line, list(scores)) for line, scores in dataset]
				elif reliability_mode:
					# try manually convert the dataset to list
					dumpable_dataset = [list(item) for item in dataset]
				else:
					dumpable_dataset = dataset
				json.dump(dumpable_dataset, raw_dataset_file, ensure_ascii=False, cls=NumpySupportedEncoder, **json_dump_kwargs)
				if(duplicate_mode):
					del dumpable_dataset
	# reduce the dataset, balance, and split if necessary
	# only one or zero options avaiable at once
	print("Normalize filtered data and split if necessary")
	assert not duplicate_mode or not reliability_mode, "modes are mutually exclusive"
	if(eval_set_size is not None):
		print("Dataset pre-splitted, do not balance")
	elif(duplicate_mode):
		dataset = balanceSetByPossibleElement(dataset, debug=debug, certainty_score=extended_output)
		# after balancing, reduce anyway
		dataset = balanceSetByReduction(dataset, debug=debug, reduce_mode=reduce_mode)
	elif(reliability_mode):
		dataset = balanceSetByAllScores(dataset, debug=debug, certainty_score=extended_output, reduce_mode=reduce_mode)
	else:
		dataset = balanceSetByReduction(dataset, debug=debug, reduce_mode=reduce_mode)
	
	# eval is min(1000, 10% set)
	distrib_graph_location = os.path.splitext(save_location)[0] + "_distribution.png"
	if(split_train_and_eval):
		if(eval_set_size is None):
			# not predetermined split, decide on threshold and shuffle and split
			eval_set_size = min(1000, len(dataset) // 10)
			shuffle(dataset)
		if(monolingual_file is not None):
			# because eval is taking from the front, the eval should not be affected
			print("Monolingual file detected, adding extra sentences as 3")
			dataset = addMonolingualAsNeutral(dataset, monolingual_file)
		split_data = {
			"train": dataset[eval_set_size:], 
			"eval": dataset[:eval_set_size],
			"data_is_extended": extended_output
		}
		with io.open(save_location, "w", encoding="utf-8") as save_file:
			print("Dumping splitted dataset @ {}".format(save_location))
			json.dump(split_data, save_file, ensure_ascii=False, cls=NumpySupportedEncoder, **json_dump_kwargs)
		with io.open(distrib_graph_location, "wb") as graph_file:
			print("Graph splitted dataset @ {}".format(distrib_graph_location))
			createDistributionGraph(graph_file, [split_data["train"], split_data["eval"], dataset], "Data distribution (Split)", legend_names=["train", "eval", "total"])
		return split_data
	else:
		with io.open(save_location, "w", encoding="utf-8") as save_file:
			print("Dumping un-splitted dataset @ {}".format(save_location))
			json.dump(dataset, save_file, ensure_ascii=False, cls=NumpySupportedEncoder, **json_dump_kwargs)
		with io.open(distrib_graph_location, "wb") as graph_file:
			print("Graph un-splitted dataset @ {}".format(distrib_graph_location))
			createDistributionGraph(graph_file, [dataset], "Data distribution (Whole)", legend_names=["total"])
		return dataset
