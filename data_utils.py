import collections, io
import numpy as np

import scripts.cleaner as cleaner
import xer

PAD_TOKEN = "<pad>"
START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def build_word_dict(filename):
	with io.open(filename, "r", encoding="utf-8") as f:
		words = f.read().replace("\n", "").split()

	word_counter = collections.Counter(words).most_common()
	word_dict = dict()
	word_dict[PAD_TOKEN] = 0
	word_dict[START_TOKEN] = 1
	word_dict[END_TOKEN] = 2
	for word, _ in word_counter:
		word_dict[word] = len(word_dict)

	return word_dict


def build_dataset(filename, word_dict):
	with io.open(filename, "r", encoding="utf-8") as f:
		lines = f.readlines()
		data = list(map(lambda s: s.strip().split(), lines))

	unk_id = word_dict[UNK_TOKEN]
	max_document_len = max([len(s) for s in data]) + 2
	data = list(map(lambda s: [START_TOKEN] + s + [END_TOKEN], data))
	data = list(map(lambda s: [word_dict.get(w, unk_id) for w in s], data))
	data = list(map(lambda d: d + (max_document_len - len(d)) * [word_dict[PAD_TOKEN]], data))

	return data


def batch_iter(inputs, batch_size, num_epochs):
	inputs = np.array(inputs)

	num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
	for epoch in range(num_epochs):
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, len(inputs))
			yield inputs[start_index:end_index]


def loadWords(word_file):
	word_set = set()
	for c_word in word_file.readlines():
		c_word = c_word.strip().lower()
		if(" " in c_word):
			word_set.update(c_word.split())
#			if(all( (len(subword)==1 for subword in c_word.split()) )):
#				print("Problem with word {:s}, splitted {}".format(c_word, c_word.split()))
		else:
			word_set.add(c_word)
	return word_set

# subtitute weight, insert weight, delete weight
# TODO a customized xer for word lacking in diacritics
DEFAULT_DISTANCE_WEIGHT = (2, 2, 2)
def createCalculateDistance(weights=DEFAULT_DISTANCE_WEIGHT):
	sub_weight, ins_weight, del_weight = weights
	def calculateDistance(word1, word2):
		# the sub-ins-del ops are from word2 to word1
		subtitution, insertion, deletion = xer.levenshtein(word1, word2, sub_cost=sub_weight, add_cost=ins_weight, del_cost=del_weight)[-1]
		return subtitution + insertion + deletion
	return calculateDistance

def createCalculateDistanceWithDiacritics():
	ins_weight = del_weight = 3
	_, transform_dict = cleaner.generate_tranform_dict()
	# all subtitution toward adding diacritics are 1
	# set of add 1 diac
	lvl1_set = set()
	for sub_dict in transform_dict.values():
		lvl1_set.update(sub_dict.items())
	# set of add 2 diac
	lvl2_set = set()
	for org, tar in lvl1_set:
		for sub_dict in transform_dict.values():
			if(tar in sub_dict):
				lvl2_set.add( (org, sub_dict[tar]) )
	def modified_sub_cost(w1, w2):
		# 3 for totally unrelated word, 1/2 for those in previous dict
		if(w1 != w2):
			# remember, word2 to word1
			searcher_tuple = (w1, w2)
			if(searcher_tuple in lvl1_set):
#				print("lvl1 found: {}-{}".format(w2, w1))
				return 1
			elif(searcher_tuple in lvl2_set):
#				print("lvl2 found: {}-{}".format(w2, w1))
				return 2
			else:
				return 3
		else:
			return 0
	def calculateDistance(word1, word2):
		# the sub-ins-del ops are from word2 to word1
		subtitution, insertion, deletion = xer.levenshtein(word1, word2, sub_cost=modified_sub_cost, add_cost=ins_weight, del_cost=del_weight)[-1]
		return subtitution + insertion + deletion
	return calculateDistance

def getCloselyRelatedWords(word_set, word_list=None, relate_range=5):
	word_list = list(word_set) if word_list is None else word_list
	relation = {}
	distance_fn = createCalculateDistanceWithDiacritics()
	for word in word_list:
		word_relation = ((c_word, distance_fn(word, c_word)) for c_word in word_set if word != c_word)
		word_relation = [word_and_distance for word_and_distance in sorted(word_relation, key=lambda item: item[-1]) if word_and_distance[-1] <= relate_range]
		relation[word] = word_relation
		print("Word {:s}: {}".format(word, word_relation[:10]))
	return relation
