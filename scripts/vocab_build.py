import sys, io, re
from string import punctuation
#from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def remove_words(texts, splitter_fn):
	new_texts = []
	for text in texts:
		pieces = (p for p in splitter_fn(text) if p != "")
		new_texts.extend(pieces)
	del texts
	return new_texts

def debug_checker(item):
	if(len(item) != 4):
		print("Item error: {}".format(item))
		return False
	else:
		return True

if __name__ == "__main__":
	data_file, vocab_file = sys.argv[1:3]
	with io.open(data_file, "r", encoding="utf-8") as data:
		data = (line.strip().split("\t") for line in data.readlines())
		checked_data = (line for line in data if debug_checker(line))
		_, _, _, texts = zip(*checked_data)
		texts = list(texts)
		print("Data loaded, number of samples:", len(texts))
	
#	with io.open("/home/quan/Workspace/Source/neural_network/vietnamese-stopwords.txt", "r", encoding="utf-8") as stop_word_file:
#		stop_words = [line.strip() for line in stop_word_file]
#		stop_word_detector = re.compile("|".join(stop_words))
	number_detector = re.compile("[\d\.\,]+?")
	# remove the underscore from the punctuation, as well as adding the stranger tokens
	punctuation = punctuation.replace("_", "") + "“”"
	punctuation_detector = re.compile("[{}]+".format(punctuation))
	
	texts = remove_words(texts, lambda text: re.split(number_detector, text))
	print("Number removed, texts:", len(texts))
	texts = remove_words(texts, lambda text: re.split(punctuation_detector, text))
	print("Punctuation removed, texts:", len(texts))
#	texts = remove_words(texts, lambda text: re.split(stop_word_detector, text))
#	print("Stopwords removed, texts:", len(texts))
	unigram_counter = Counter( (word for text in texts for word in text.split()) )
	print("1-Counter size", len(unigram_counter))
	print("1-Counter least_common", unigram_counter.most_common()[-10:])
	print("1-Counter highest_underscore:", next((item for item, count in unigram_counter.most_common() if "_" in item)) )
	# top the vocab into words that had appeared more than 5 times
	unigram_vocab = {word:idx for idx, (word, occurence) in enumerate(unigram_counter.most_common()) if occurence >= 5}
	print("1-Counter occ>5 size", len(unigram_vocab))
	# further split the texts and convert them into list of indices
	ids_texts = []
	for text in texts:
		text_to_ids = (unigram_vocab.get(word, -1) for word in text.strip().split())
		piece = []
		for idx in text_to_ids:
			if(idx != -1):
				piece.append(idx)
			elif(len(piece) == 0):
				pass
			else:
				ids_texts.append(piece)
				piece = []
			if(len(piece) > 0):
				ids_texts.append(piece)
	print("Converted to ids, size: ", len(ids_texts))
	# delete the text to ready for the counter
	del texts
	# extract n-gram key from the indiced texts
	ngram_keys = lambda ids_text, ngram: (tuple(ids_text[i:i+ngram]) for i in range(len(ids_text)) if i+ngram<=len(ids_text))
	bigram_counter = Counter( key for ids_text in ids_texts for key in ngram_keys(ids_text, 2) )
	print("2-Counter size", len(bigram_counter))
	print("2-Counter 10 most_common", bigram_counter.most_common()[-10:])
	unigram_reverse_vocab = {idx:word for word, idx in unigram_vocab.items()}
	bigram_from_ids = lambda ngram: " ".join([unigram_reverse_vocab[i] for i in ngram])
	with io.open(vocab_file, "w", encoding="utf-8") as vocab:
		bigram_phrases = ( (bigram_from_ids(ngram), count) for ngram, count in bigram_counter.most_common())
		unigram_phrases = iter(unigram_counter.most_common())
		bigram, bigram_count = next(bigram_phrases)
		unigram, unigram_count = next(unigram_phrases)
		for i in range(100000):
			# write the higher pieces and draw them from the phrases
			if(bigram_count > unigram_count):
				vocab.write("{}\t{}\n".format(bigram, bigram_count))
				bigram, bigram_count = next(bigram_phrases, (None, -1))
			else:
				vocab.write("{}\t{}\n".format(unigram, unigram_count))
				unigram, unigram_count = next(unigram_phrases, (None, -1))
		print("Vocab file written to {}".format(vocab_file))
