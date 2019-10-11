#!/usr/bin/python

import sys, getopt, re, time
CALCULATE_TIME = False
SORT_LIST_DURING_FIND = False
STANDARD_CYCLE = 10000
START_PHRASES = 0
FULL_SEARCH = True

## Script deal with languages with whitespaces.
def main(argv):
	inputfile1 = None
	inputfile2 = None
	optionalfile = None
	outputfile = None
	washoutPercentage = 0.2
	phrasesMaxLength = 6
	start_time = time.time()
	try:
		opts, args = getopt.getopt(argv,"hts1:2:o:p:w:l:",["i1=","i2=","ifile1=","ifile2=","ofile=","opfile=","washout=","phlen=","time","help","sort"])
	except getopt.GetoptError:
		print('Error getting arguments')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('arguments: -i1 <inputfile1> -i2 <inputfile2> -o <outputfile>')
			print('           -op <optionalfile> -w <washout> -pl <phrasesMaxLength>')
			sys.exit()
		elif opt in ("-1", "-i1", "--ifile1"):
			try: 
				inputfile1 = open(arg, "r", encoding="utf-8")
			except IOError:
				print('Cannot read file ', arg)
				sys.exit()
		elif opt in ("-2", "-i2", "--ifile2"):
			try: 
				inputfile2 = open(arg, "r", encoding="utf-8")
			except IOError:
				print('Cannot read file ', arg)
				sys.exit()
		elif opt in ("-o", "--ofile"):
			try: 
				outputfile = open(arg, "w", encoding="utf-8")
			except IOError:
				print('Cannot read file ', arg)
				sys.exit()
		elif opt in ("-w", "--washout"):
			arg = float(arg)
			if(washoutPercentage > 0 and washoutPercentage < 1):
				washoutPercentage = arg
			else:
				print("Washout value should be in the range [0-1], defaulting")
		elif opt in ("-l", "-pl", "--phlen"):
			arg = int(arg)
			if(arg > 1):
				phrasesMaxLength = arg
		elif opt in ("-t", "--time"):
			global CALCULATE_TIME
			CALCULATE_TIME = True
		elif opt in ("-s", "--sort"):
			global SORT_LIST_DURING_FIND
			SORT_LIST_DURING_FIND = True
	if(inputfile1 is None or inputfile2 is None or outputfile is None): 
		print('Missing one of i1/i2/o arguments.')
		sys.exit()
		
	firstDict = {}
	secondDict = {}
	
	firstArray = indexingSentences(inputfile1)
	secondArray = indexingSentences(inputfile2)
	
	# singleWord = findWordsWithDict(START_PHRASES, firstArray, [], defaultWashout(START_PHRASES))
	# singleWord.sort(key=lambda x: x.listLength, reverse=True)
	# wordListByLength.append(singleWord)
	wordListByLength = []
	for i in range(START_PHRASES, phrasesMaxLength):
		if(FULL_SEARCH):
			bare = findWordsBare(i+1,firstArray, defaultWashout(i+1))
			wordListByLength.append(bare)
			firstDict.update(bare)
		else:
			wwd = findWordsWithDict(i+1, firstArray, wordListByLength[i-START_PHRASES] if i != START_PHRASES else {}, defaultWashout(i+1))
			wordListByLength.append(wwd)
			firstDict.update(wwd)
	print('Loop done for source')
	
	wordListByLength = []
	for i in range(START_PHRASES, phrasesMaxLength):
		if(FULL_SEARCH):
			bare = findWordsBare(i+1,secondArray, defaultWashout(i+1))
			wordListByLength.append(bare)
			secondDict.update(bare)
		else:
			wwd = findWordsWithDict(i+1, secondArray, wordListByLength[i-START_PHRASES] if i != START_PHRASES else {}, defaultWashout(i+1))
			wordListByLength.append(wwd)
			secondDict.update(wwd)
	print('Loop done for target')
	
	timer = time.time()
	source = list(map(lambda sen: tokenize(sen,firstDict,phrasesMaxLength), firstArray))
	target = list(map(lambda sen: tokenize(sen,secondDict,phrasesMaxLength), secondArray))
	correlation = addCorrelation(source, target, 6)
	comparingCor = addCorrelation(target, source, 10)
	
	validDict = validityCheck(correlation, comparingCor, (firstDict, secondDict, 5))
	revValidDict = validityCheck(comparingCor, correlation)
	removeValidResult(correlation, revValidDict)
	if(CALCULATE_TIME):
		print("All mapping cost %ss" % (time.time() - timer))
	
	outputfile.write("###Valid set found###\n\n\n")
	for key in validDict:
		outputfile.write("{}: [{}]\n".format(key,validDict[key]))
		correlation.pop(key, None)
	
	outputfile.write("\n\n\n###Other set found###\n\n\n")
	for key in correlation:
		writeToFile = True
		ws = key + ": "
		if(len(correlation[key]) == 0):
			# ws += "No correlation found, consider lowing the threshold."
			writeToFile = False
		elif(len(correlation[key][0]) == 2):
			for phr,num in correlation[key]:
				ws += "[{}][{}] ".format(phr, num)
		elif(len(correlation[key][0]) == 3):
			for phr,num,total in correlation[key]:
				ws += "[{}][{}/{}] ".format(phr, num, total)
		else:
			ws += "Wrong type in file. please recheck."
		ws += '\n'
		if(writeToFile):
			outputfile.write(ws)
	
	outputfile.write("\n\n\n###Extra (ReverseDict)###\n\n\n")
	for key in revValidDict:
		outputfile.write("{}: [{}]\n".format(key,revValidDict[key]))
		correlation.pop(key, None)
	"""
	for idx in range(len(source)):
		sen = source[idx]
		ws = "[" + "][".join(sen) + "]\n"
		if(idx < len(target)):
			sen = target[idx]
			ws += "= [" + "][".join(sen) + "]\n"
		else:
			ws += "= Cannot find sentence"
		outputfile.write(ws)
		
	for dict in reversed(wordListByLength):
		if(not FULL_SEARCH):
			dict = {k : len(v) for k,v in dict.items()}
		for key in dict:
			try:
				outputfile.write("{} - occurred {}\n".format(key, dict[key]))
			except Exception as e:
				print('Error {} while writing set {} - {}'.format(e.errno,key,dict[key]))
	"""
	if(CALCULATE_TIME):
		print("--- %s seconds --- in total" % (time.time() - start_time))
	
	inputfile1.close()
	inputfile2.close()
	outputfile.close()

# Sentences are packed into an array, split by dot/comma/whatever the hell it is and lowercased
# Normalize later
def indexingSentences(file):
	timer  = time.time()
	array = []
	content = file.readlines()
	# content = filter(lambda x: not re.match(r'^\s*$', x), content)
	for st in content:
		# st = st.strip("\n")
		st = st.strip("\n").replace(" ,", "").lower()
		if(st.find("&amp;") > 0 or st.find("&quot;") > 0 or st.find("&apos;") > 0):
			st = st.replace("&amp;",";").replace("&quot;","\"").replace("&apos;","\'")
			# print('special char detected, sentence: {}'.format(st))
		#if("." in st or "\""):
			# print('mult sentence detected, split: {}'.format(re.split("\.|,|\"|;|:", st)))
			# st = [splitted.strip( ).lower() for splitted in re.split("\.|\"", st)] # |,|;|:
			#array.extend(filter(lambda x: not re.match(r'^\s*$', x), st))
		#else:
		array.append(st)
	if(CALCULATE_TIME):
		print("Read completed, cost %s" % (time.time() - timer))
	return array
	

# Tokenize sentence from dict
def tokenize(sentence, dict, maxSearchLen):
	words = sentence.split(' ')
	
	i = 0
	j = 0
	result = []
	while(i < len(words)):
		j = i + maxSearchLen if i + maxSearchLen < len(words) else len(words)
		while(i < j):
			key = ' '.join(words[i:j])
			if key in dict:
				result.append(key)
				break
			j -= 1
		if(i == j):
			#if(words[i] not in dict):
			#	print("Cannot find single word {}.".format(words[i]))
			result.append(words[i])
			i += 1
		else:
			i = j
	return result

def addCorrelation(sourceTokens, resultTokens, maxPreserve=None, percentageThreshold=None, targetDict=None):
	timer = time.time()
	correlation = {}
	for idx in range(len(sourceTokens)):
		sourceSen = sourceTokens[idx]
		resultSen = resultTokens[idx]
		for phrase in sourceSen:
			if(phrase not in correlation):
				correlation[phrase] = {}
			for res in resultSen:
				correlation[phrase][res] = correlation[phrase].get(res, 0) + 1
	
	if(maxPreserve is not None):
		for key in correlation:
			correlation[key].pop('', None)
			correlation[key].pop('.', None)
			correlation[key].pop(',', None)
			listAvailable = sorted(correlation[key].items(), key=lambda x:x[1], reverse=True)[:maxPreserve]
			# listAvailable = [ for k in list]
			correlation[key] = listAvailable
	elif(percentageThreshold is not None and targetDict is not None):
		def checkKey(key):
			if key in targetDict:
				return targetDict[key]
			else:
				print("Key {} not found?\n".format(key))
				return -1
		for key in correlation:
			correlation[key].pop('', None)
			correlation[key].pop('.', None)
			correlation[key].pop(',', None)
			listAvailable = [(k, correlation[key][k], targetDict.get(k, -1)) for k in correlation[key] if (correlation[key][k] / targetDict.get(k, -1)) >= percentageThreshold ]
			# listAvailable = [ for k in list]
			correlation[key] = listAvailable
	correlation.pop(' ', None)
	correlation.pop('', None)
	correlation.pop(',', None)
	correlation.pop('.', None)
	if(CALCULATE_TIME):
		print("Correlation ran, cost {}, creating {} keys".format(time.time() - timer, len(correlation)))
	return correlation

# Checker is tuple (sourceDict, targetDict, threshold)
def validityCheck(currentDict, otherDict, checker=None):
	timer = time.time()
	
	result = {}
	
	for key in currentDict:
		if(len(currentDict[key]) == 0):
			continue
		for val in currentDict[key]:
			checkKey = val[0]
			if(checkCorrelation(otherDict, checkKey, key)):
				# Found in the other dict
				if(checker is None or (checker[0].get(key, 0) > checker[2] and checker[1].get(checkKey, 0) > checker[2])):
					result[key] = checkKey
				break
	
	if(CALCULATE_TIME):
		print("Validity ran, cost {}, verified {} keys".format(time.time() - timer, len(result)))
	return result

def removeValidResult(correlation, validPairs):
	for key in correlation:
		listValues = correlation[key]
		correlation[key] = list(filter(lambda tup: tup[0] not in validPairs, listValues))

def checkCorrelation(dict, key, searchKey):
	if(key not in dict):
		return False
	for val in dict[key]:
		if(val[0] == searchKey):
			# print("Found {} for key {}, val {}".format(searchKey, key, val[1]))
			return True
	return False

# Use dict to find phrase
def findWordsWithDict(length, sentenceArray, prevDict, threshold):
	newWordList = {}
	timer = time.time()
	passingNumber = threshold
	
	print('FindWordsArguments: length {}, sentenceArray {}, prevDict {}, threshold {}'.format(length, len(sentenceArray), len(prevDict), threshold))
	
	if(length > START_PHRASES):
		for oldPhrase in prevDict:
			if(length >= 4):
				print("oldPhrase {}".format(oldPhrase))
			if(not isinstance(threshold, int) and not threshold.is_integer()):
				passingNumber = threshold * float(oldPhrase.listLength)
			possibleNextWord = []
			#phraseTimer = time.time()
			for senIdx in prevDict[oldPhrase]:
				if(length >= 4):
					print("addWordValue arguments phrase {} sentenceIdx {}".format(oldPhrase, senIdx))
				possibleNextWord = addWordValue(possibleNextWord, sentenceArray[senIdx], oldPhrase, senIdx)
			#print('word: {}, amount of possibleNextWord: {}'.format(oldPhrase, len(possibleNextWord)))
			for nextWord in possibleNextWord:
				if(nextWord.listLength >= passingNumber):
					nextWord.phrase = oldPhrase + ' ' + nextWord.phrase
					newWordList[nextWord.phrase] = nextWord.listSentences
					if(CALCULATE_TIME):
						print("Add new word {} {} = {}".format(nextWord.phrase, len(nextWord.listSentences), len(newWordList[nextWord.phrase])))#, time.time() - phraseTimer))
		
	elif(length == START_PHRASES):
		counter = 1
		loopTimer = time.time()
		for st in sentenceArray:
			if(counter % STANDARD_CYCLE == 0):
				if(CALCULATE_TIME):
					print('finished {}/{} lines, time cost {}'.format(counter, len(sentenceArray),time.time() - loopTimer))
					loopTimer = time.time()
			counter += 1
			senIdx = sentenceArray.index(st)
			for word in createWordList(st, length):
				if(word not in newWordList):
					newWordList[word] = []
				newWordList[word].append(senIdx)
		# pruning after
		newWordList = {k: v for k, v in newWordList.items() if len(v) > passingNumber}
	else:
		sys.exit()
	print("Loop {} handled {} old phrases, created {} new phrases, cost {}".format(length, len(prevDict), len(newWordList), time.time() - timer))
	return newWordList
	
def findWordsBare(length, sentenceArray, threshold):
	newWordList = {}
	timer = time.time()
	counter = 1
	loopTimer = time.time()
	for st in sentenceArray:
		if(counter % STANDARD_CYCLE == 0):
			if(CALCULATE_TIME):
				print('finished {}/{} lines, time cost {}'.format(counter, len(sentenceArray),time.time() - loopTimer))
				loopTimer = time.time()
		counter += 1
		for word in createWordList(st, length):
			if(word not in newWordList):
				newWordList[word] = 0
			newWordList[word] += 1
	# pruning after
	newWordList = {k: v for k, v in newWordList.items() if v > threshold}
	if(CALCULATE_TIME):
		print("Loop {} created {} new phrases (passing {}), cost {}".format(length, len(newWordList), threshold, time.time() - timer))
	return newWordList
	
def createWordList(sentence, phraseLen):
	splitter = sentence.split()
	wordList = []
	if(phraseLen == 1):
		return splitter
	for i in range(len(splitter) - phraseLen + 1):
		wordList.append(' '.join(splitter[i:i+phraseLen]))
	return wordList

## March values from an ordered array
## Unused for now
def getMatchingValuesInOrderedArray(first,second):
	i = len(first)
	j = len(second)
	array = []
	while(i >= 0 and j >= 0):
		if(first[i] == second[j]):
			array.insert(0, first[i])
			i -= 1
			j -= 1
		elif(first[i] > second[j]):
			i -= 1
		else:
			j -= 1
	return array

## changes word value
def addWordValue(wordArray, sentence, phrase, sentenceIdx):
	idx = sentence.find(phrase)
	word = ''
	available = False
	while(idx >= 0):
		idx += len(phrase) + 1
		if(idx < len(sentence)):
			# phrase not at end of file
			wordEnd = sentence.find(' ',idx)
			if(wordEnd < 0):
				# next is final word, hence no whitespace
				word = sentence[idx:]
			else:
				word = sentence[idx:wordEnd]
			# add word count or make new if not existing
			wordExist = next((x for x in wordArray if x.phrase == word), None)
			if(wordExist is None):
				wordExist = Phrase(word)
				wordArray.append(wordExist)
			wordExist.addSentence(sentenceIdx)
			
		idx = sentence.find(phrase, idx)
	return wordArray

def defaultWashout(tier):
	return {
		1: 10,
		2: 200,
		3: 100,
		4: 20,
		5: 20,
		6: 10,
		7: 10,
		8: 10,
		9: 10,
		10: 10
	}.get(tier, 10)
	
class SentenceCouple:
	sentence1 = ''
	sentence2 = ''

	def __init__(self, s1, s2):
		self.sentence1 = s1
		self.sentence2 = s2
	
	def haveWord(self, wordIn, word):
		if((wordIn == 1 and self.sentence1.find(word) != -1) or (wordIn == 2 and self.sentence2.find(word) != -1)):
			return True
		else:
			return False
	
class Phrase:
	def __init__(self, ph):
		self.phrase = ph
		self.listLength = 0
		self.listSentences = []
	
	def addSentence(self,input):
		if(isinstance(input, int) or input.is_integer()):
			self.listSentences.append(input)
			self.listLength += 1
		elif(isinstance(input, list)):
			self.listSentences.extend(input)
			self.listLength += len(input)
	
	
if __name__ == "__main__":
	main(sys.argv[1:])