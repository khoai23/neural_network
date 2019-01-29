#import tensorflow as tf
import scripts.cleaner as cleaner
import io, xer
import argparse

WORD_FILE_LOCATION = "/home/quan/Workspace/Source/translator/txt/Viet74K.txt"
WORD_RELATION_DUMP = "/home/quan/Workspace/Source/translator/txt/xer_distance.txt"

START_OF_SENTENCE_TOKEN = "<s>"
END_OF_SENTENCE_TOKEN = "<\s>"

def convertSentenceToSkipgram(line, window_size=2):
	# assume line is properly tokenized already
	tokens = [START_OF_SENTENCE_TOKEN] * window_size + line.strip().split() + [END_OF_SENTENCE_TOKEN] * window_size
	full_window_size = 2 * window_size + 1
	[ tokens[i:i+full_window_size] for i in range(len(tokens) - window_size * 2)]

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Language Models.')
	parser.add_argument('-m', '--model_type', type=str, choices=["skipgram", "whole"], default="skipgram", help='The type of model used')
	args = parser.parse_args()
	
	with io.open(WORD_FILE_LOCATION, "r", encoding="utf-8") as word_file:
		word_set = loadWords(word_file)
		print("Word set loaded with {:d} elements".format(len(word_set)))
	
	word_relation = getCloselyRelatedWords(word_set, word_list=["toi", "chao", "ban"], relate_range=5)

