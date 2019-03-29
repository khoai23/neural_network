import io, os, json, sys, re
from scripts.cleaner import generate_tranform_dict, vietnamese_ghost_characters_cleaner
import argparse
from sentiment_data import _loadJSONConfig

CSV_PATH = "/home/quan/Workspace/Data/espresso/VnEmoLex.csv"
TXT_PATH = "/home/quan/Workspace/Data/espresso/VietSentiWordnet_Ver1.3.5.txt"
REV_PATH = "/home/quan/Workspace/Data/espresso/NegDict.txt"
def loadCSV(path):
	with io.open(path, "r", encoding="utf-8") as csv_file:
		negative_set, positive_set = set(), set()
		for line in csv_file.readlines():
			list_item = line.split(",")
			if(len(list_item) < 4):
				print("line not in format: {:s}".format(line))
				continue
			orig, true_phrase, pos, neg, *_ = list_item
			true_phrase = true_phrase.strip().replace("\"", "")
			try:
				pos = int(pos)
				neg = int(neg)
			except ValueError:
				print("Error when converting {} and {} to int".format(pos, neg))
				continue
			if(pos == 1):
				positive_set.add(true_phrase)
			elif(neg == 1):
				negative_set.add(true_phrase)
			else:
				print("Phrase {:s} not positive/negative".format(true_phrase))
	return positive_set, negative_set

def loadTXT(path, cleaner_fn=None):
	with io.open(path, "r", encoding="utf-8") as txt_file:
		negative_set, positive_set = set(), set()
		for line in txt_file.readlines():
			if(line[0] == "#"):
				# first header line, ignore
				continue
			_, _, pos_score, neg_score, term, _ = line.strip().split("\t")
			# find all terms that is in the set, separated by space
			terms = term.strip().split() if term.find(" ") >= 0 else [term]
			# term is accompanied by #{num}, get rid of it and remove underscore
			true_term = [ t.split("#")[0].replace("_", " ") for t in terms ]
			if(cleaner_fn):
				true_term = [ cleaner_fn(t) for t in true_term ]
			# convert to float
			pos_score, neg_score = float(pos_score), float(neg_score)
			if(pos_score > neg_score):
				positive_set.update(true_term)
			elif(pos_score < neg_score):
				negative_set.update(true_term)
			else:
				print("Phrase(s) {} have equal pos/neg: {}={}".format(true_term, pos_score, neg_score))
		return positive_set, negative_set

def loadReversalPhrases(path=None, cleaner_fn=lambda item: item):
	if(path is not None):
		with io.open(path, "r", encoding="utf-8") as txt_file:
			# remove underscore and endline
			return set( (cleaner_fn(phr.strip().replace("_", " ")) for phr in txt_file.readlines()) )
	else:
		return {"không","chẳng","k","ko","chả","không thể", "không có"}

SAVE_PATH = "/home/quan/Workspace/Data/espresso/dict.json"
def loadDictionary(save_path=SAVE_PATH, dict_paths=[CSV_PATH], rev_paths=[REV_PATH], force_reload=False):
	if(os.path.isfile(save_path) and not force_reload):
		with io.open(save_path, "r", encoding="utf-8") as save_file:
			data = json.load(save_file)
			data = [set(item) for item in data]
	else:
		# prepare cleaner
		all_ghost_diacritics, convert_dict = generate_tranform_dict()
		ghost_checker_regex = re.compile("[{:s}]".format(all_ghost_diacritics))
		cleaner_func = lambda phr: vietnamese_ghost_characters_cleaner(phr, all_ghost_diacritics, convert_dict, ghost_checker_regex=ghost_checker_regex)
		# read the positive/negative from files
		positive_set, negative_set = set(), set()
		for dict_path in dict_paths:
			if(dict_path[-4:] == ".csv"):
				print("reading file in CSV mode")
				pos_set, neg_set = loadCSV(dict_path)
				positive_set.update(pos_set)
				negative_set.update(neg_set)
			elif(dict_path[-4:] == ".txt"):
				print("reading file in TXT mode")
				pos_set, neg_set = loadTXT(dict_path, cleaner_fn=cleaner_func)
				positive_set.update(pos_set)
				negative_set.update(neg_set)
		reverse_set = set()
		# read the reversal from files
		for rev_path in rev_paths:
			reverse_set.update( loadReversalPhrases(rev_path, cleaner_fn=cleaner_func) )
		# update, remove all the wrong versions
		positive_set, negative_set = remakeSets(positive_set, negative_set, reverse_set)
		with io.open(save_path, "w", encoding="utf-8") as save_file:
			data = positive_set, negative_set, reverse_set
			data = [list(item) for item in data]
			json.dump(data, save_file, ensure_ascii=False)
	return data

def remakeSets(positive_set, negative_set, reverse_set):
	# move all reversed pieces in positive/negative to their correct set
	# phrases must have _ to prevent reading inword (e.g k_ in kho)
	reverse_phrases = [(phr + " ") for phr in sorted(reverse_set, key=lambda it: len(it), reverse=True)]
	repeller_set = set()
	for item in positive_set:
		if(item.find(" ") < 0):
			# one word will be exempted
			continue
		if(any( (phr == item[:len(phr)] for phr in reverse_phrases) )):
			print("Reversed phrase detected(pos): {:s}".format(item))
			cut_length = next( (len(phr) for phr in reverse_phrases if phr in item) )
			repeller_set.add(item)
			negative_set.add(item[:cut_length].strip())
	positive_set = positive_set.difference(repeller_set)
	# do the same thing for negative
	repeller_set = set()
	for item in negative_set:
		if(item.find(" ") < 0):
			# one word will be exempted
			continue
		if(any( (phr == item[:len(phr)] for phr in reverse_phrases) )):
			print("Reversed phrase detected(neg): {:s}".format(item))
			repeller_set.add(item)
			positive_set.add(item[:cut_length].strip())
	negative_set = negative_set.difference(repeller_set)
	return positive_set, negative_set

def constructSearchDictionary(data):
	# return dictionary: 1 is positive, 0 is reverse, -1 is negative, max_length
	positive_set, negative_set, reverse_set = data
	full_dict, max_length = {}, 0
	for item in positive_set:
		full_dict[item] = 1
		if(len(item.split()) > max_length):
			max_length = len(item.split())
	for item in negative_set:
		full_dict[item] = -1
		if(len(item.split()) > max_length):
			max_length = len(item.split())
	for item in reverse_set:
		full_dict[item] = 0
		if(len(item.split()) > max_length):
			max_length = len(item.split())
	return full_dict, max_length

def tryApplyDictionary(sentence, dictionary, reverse_range=2):
	# find all phrases available on the sentence and try to see if it conform to the rating
	# TODO positive may have duplicate? nhanh vs nhanh hon?
	full_set, max_length = dictionary
	tokens = sentence.strip().split() if sentence.strip().find(" ") > 0 else [sentence]
	# iterate through the sentence, checking for phrases
	phrases_found, last_phrase_ended = [], 0
	for i in (range(0, len(tokens) - max_length) if max_length < len(tokens) else [0]):
		if(i < last_phrase_ended):
			# ignore overlap
			continue
		for phr_len in reversed(range(2, max_length+1)):
			# check for all phrases starting with token i and with length <=maxlengh
			phrase = " ".join(tokens[i:i+phr_len])
			if(phrase in full_set):
#				print("Found phrase: {:s}, value: {:d}".format(phrase, full_set[phrase]))
				phrases_found.append( (i, phrase, full_set[phrase]) )
				last_phrase_ended = i+phr_len
				break
		if(last_phrase_ended <= i and tokens[i] in full_set):
				phrase = tokens[i]
#				print("Found phrase: {:s}, value: {:d}".format(phrase, full_set[phrase]))
			# the previous process do not found any value, and the token is in 
				phrases_found.append( (i, phrase, full_set[phrase]) )
				last_phrase_ended = i+1
	# reverse all the phrases after a reverse
	ignore = -1
	merged = []
	for i, (index, phrase, value) in enumerate(phrases_found):
		if(index <= ignore):
			continue
		else:
			if(value == 0):
				# do reversing if the distance to the next is <= reverse_range and is not a reverse phrase itself
				# first is for the phrase to exist
				if(i < len(phrases_found)-1 and phrases_found[i+1][0] < index + reverse_range and phrases_found[i+1][-1] != 0):
					rev_idx, rev_phr, rev_val = phrases_found[i+1]
					ignore = rev_idx
					merged.append( (index, " ".join((phrase, rev_phr)), -rev_val))
				else:
#					print("Orphaned reverse phrase: {:s}".format(phrase))
					pass
			else:
				# append as normal
				merged.append( (index, phrase, value) )
#	print("Converted from {} -> {}".format(phrases_found, merged))
	return merged

def debugDictCoverage(sentences, dictionary):
	true_dict = constructSearchDictionary(dictionary)
	coverage_count = sum( (1 if len(tryApplyDictionary(sentence, true_dict)) > 0 else 0 for sentence in sentences) )
	print("Coverage: {:d}/{:d}({:2.2f}%)".format(coverage_count, len(sentences), float(coverage_count) / float(len(sentences)) * 100.0))

def debugApplicationCoverage(sentences, ratings, dictionary):
	true_dict = constructSearchDictionary(dictionary)
	correct_count = 0
	neutral_count = 0
	for sentence, rating in zip(sentences, ratings):
		# check
		if(rating == 3):
			# neutral, not handle-able
			neutral_count += 1
		elif(rating == 5 or rating == 1):
			# monolithic values, with no downsides
			pref_value = (rating - 3) // 2
			caught_phrases = tryApplyDictionary(sentence, true_dict)
			if(len(caught_phrases) >= 2 and all( (phr[-1] == pref_value for phr in caught_phrases) )):
				correct_count += 1
		elif(rating == 4 or rating == 2):
			# singular rating, or conflicting
			pref_value = rating - 3
			caught_phrases = tryApplyDictionary(sentence, true_dict)
			if((len(caught_phrases) == 1 and caught_phrases[0][-1] == pref_value) or any( (phr[-1] == -pref_value for phr in caught_phrases) )):
				correct_count += 1
		else:
			neutral_count += 1
			print("WTF rating value??? : {}".format(rating))
	total_count = len(sentences) - neutral_count
	print("Apply correct: {:d}/{:d}({:2.2f}%)".format(correct_count, total_count, float(correct_count) / float(total_count) * 100.0))

def tryApplyDictionaryToPredictions(sentences, ratings, dictionary, debug=False):
	true_dict = constructSearchDictionary(dictionary)
	true_ratings = []
	for sentence, rating in zip(sentences, ratings):
		caught_phrases = tryApplyDictionary(sentence, true_dict)
		if(rating == 1 or rating == 5):
			# check if there is opposite phrases, if yes, reduce them
			positivity = (rating - 3) // 2
			if(any( (phr[-1] == -positivity for phr in caught_phrases) )):
#				print("Found conflicting value for {:d}, reducing.".format(rating))
				true_ratings.append(rating - positivity)
			else:
				true_ratings.append(rating)
		elif(rating == 2 or rating == 4):
			# check if there are more than one compliant phrase, if yes, increase them
			positivity = rating - 3
			if(len(caught_phrases) >= 2 and all( (phr[-1] == positivity for phr in caught_phrases) ) ):
#				print("Found monolithic value for {:d}, increasing.".format(rating))
				true_ratings.append(rating + positivity)
			else:
				true_ratings.append(rating)
		else:
			# neutral, do nothing for now
			true_ratings.append(rating)
	if(debug):
		counter_old, counter_new = dict(), dict()
		for old, new in zip(ratings, true_ratings):
			counter_old[old] = counter_old.get(old, 0) + 1
			counter_new[new] = counter_new.get(new, 0) + 1
		print("Old: {}, new: {}".format(counter_old, counter_new))
	return sentences, true_ratings

DEBUG_FILE_LOCATION = "/home/quan/Workspace/Data/model/espresso/10k_v2/block_filtered_data.json"
def jsonDataReader(file_path=DEBUG_FILE_LOCATION):
	with io.open(file_path, "r", encoding="utf-8") as json_file:
		data = json.load(json_file)
	sentences, ratings = zip(*data)
	return list(sentences), [int(list(rat.keys())[0]) for rat in ratings]

LIST_AVAILABLE_FILES, LIST_REV_PATHS = _loadJSONConfig()["sentiment_dictionary"]
# TODO list the negative & exclamation later. For now, the 

if __name__ == "__main__":
	# default arguments
	parser = argparse.ArgumentParser(prog="sentiment_dictionary", description="Create Dictionary from downloaded txt/csv file")
	parser.add_argument("-f", "--files", type=str, choices=LIST_AVAILABLE_FILES.keys(), nargs="*", help="files to be used for generating")
	parser.add_argument("--force_reload", action="store_true", help="if specified, redo the building process")
	parser.add_argument("-o", "--output", type=str, default=SAVE_PATH, help="target to save into; default to {:s}".format(SAVE_PATH))
	parser.add_argument("--debug_output_path", type=str, default=DEBUG_FILE_LOCATION, help="json project data file to use to test coverage; default to {:s}".format(DEBUG_FILE_LOCATION))
	args = parser.parse_args()
	list_files_dictionary = [ LIST_AVAILABLE_FILES[name] for name in args.files ]
	list_files_negation = [REV_PATH]

	dictionary = positive_set, negative_set, reverse_set = loadDictionary(save_path=args.output, dict_path=list_files_dictionary, rev_path=list_files_negation, force_reload=args.force_reload)
	if(os.path.isfile(args.debug_output_path)):
		sentences, ratings = jsonDataReader(file_path=args.debug_output_path)
		#debugDictCoverage(sentences, dictionary)
		debugApplicationCoverage(sentences, ratings, dictionary)
	sys.exit()
