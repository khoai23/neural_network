#!/usr/bin/python

import re, time, pickle, io, sys
from trie import TrieNode as Node

import chardet
CALCULATE_TIME = True

def correctSubtitleEncoding(filename, newFilename, encoding_from, encoding_to='UTF-8'):
	with io.open(filename, 'r', encoding=encoding_from) as fr:
		with io.open(newFilename, 'w', encoding=encoding_to) as fw:
			for line in fr:
				fw.write(line[:-1]+'\r\n')

LOG_FUNC = None
def writeToLog(*args):
	if(LOG_FUNC is None):
		print(*args)
	else:
		LOG_FUNC(*args)

def bindLoggingFunction(func):
	global LOG_FUNC 
	LOG_FUNC = func

def openFileByChardet(fileDir, openMode, logger=writeToLog):
	try:
		# file = io.open(fileDir, openMode)
		fileRead = io.open(fileDir, openMode + "b").read()
		detector = chardet.detect(fileRead)
		# file.close()
		logger('Try opening file {} with encoding {}'.format(fileDir, detector['encoding']))
		return fileRead.decode(detector['encoding'])
	except IOError as e:
		logger('Cannot read file source @{}, error {}'.format(fileDir, e))
	except UnicodeDecodeError as e:
		logger('Error during decoding @{}, error {}'.format(fileDir, e))
		return fileRead.decode(detector['encoding'], errors='ignore')
	
def addPhraseToTree(treeRoot, phraseTuple, convertToFullClass=False):
	if(not isinstance(treeRoot, Node)):
		writeToLog("Error, treeRoot not TrieNode")
	if(convertToFullClass):
		value = TranslateData(phraseTuple[0], phraseTuple[1].split('/'))
	else:
		value = phraseTuple[1]
		
	treeRoot.buildNode(phraseTuple[0], -1, value)

# Import from undefined file
def importPairFromDefaultFile(file, phraseDict=None, convertToFullClass=True):
	timer  = time.time()
	if(not isinstance(phraseDict, dict) or phraseDict is None):
		phraseDict = {}
	if(isinstance(file, str)):
		file = io.open(file, "r", encoding='utf-8')
	content = file.readlines()
	content = filter(lambda x: not re.match(r'^\s*$', x), content)
	for st in content:
		st = st.split('=', 1)
		if(len(st) < 2):
			continue
		if(st[1].find("$ERROR$") >=0):
			continue
		if(convertToFullClass):
			phraseDict[st[0]] = TranslateData(st[0], re.split('/', st[1].strip()))
		else:
			phraseDict[st[0]] = st[1].strip('\n').strip()
	if(CALCULATE_TIME):
		writeToLog("Read completed, cost %.2fs" % (time.time() - timer))
	return phraseDict
	
def writeToTxtFileFromDict(file, phraseDict):
	timer  = time.time()
	if(isinstance(file, str)):
		file = io.open(file, "w", encoding='utf-8')
	for it in phraseDict:
		if(isinstance(phraseDict[it], TranslateData)):
			file.write("{}={}\n".format(it, "/".join(phraseDict[it].getAllValues())))
		elif(isinstance(phraseDict[it], str)):
			file.write("{}={}\n".format(it, phraseDict[it]))
		else:
			file.write("{}=$ERROR${}\n".format(it, phraseDict[it]))
	file.close()
	if(CALCULATE_TIME):
		writeToLog("Write completed, cost %.2fs" % (time.time() - timer))
	
def importPairFromDictionaryFile(file, phraseDict=None, convertToFullClass=False, converterFunc=None):
	if(converterFunc is None):
		regexLV = re.compile("✚\[.+?\]")
		regexTT = re.compile("\[[\w\d]+?\]")
		regexSplitter = re.compile("\\n\\t\d")
		def defaultFunc(input):
			if(input.find("$ERROR$") >= 0):
				return None
			if(input.find('✚') >= 0):
				input = re.sub(regexLV, "", input)
				if(convertToFullClass):
					input = re.split(regexSplitter, input)
				else:
					input = re.sub(regexSplitter, "/", input)
				# input.replace(regexLV, "").replace(regexSplitter, "/")
			else:
				input = re.sub(regexTT, "", input)
				if(convertToFullClass):
					input = re.split(regexSplitter, input)
				else:
					input = re.sub(regexSplitter, "/", input)
				# input = input.replace(regexTT, "").replace(regexSplitter, "/")
			return input
		converterFunc = defaultFunc
	timer  = time.time()
	if(not isinstance(phraseDict, dict) or phraseDict is None):
		phraseDict = {}
	content = file.readlines()
	content = filter(lambda x: not re.match(r'^\s*$', x), content)
	for st in content:
		st = st.split('=', 1)
		if(len(st) < 2):
			continue
		if(convertToFullClass):
			phraseDict[st[0]] = TranslateData(st[0],converterFunc(st[1].strip('\n').split('/')))
		else:
			phraseDict[st[0]] = converterFunc(st[1])
	if(CALCULATE_TIME):
		writeToLog("Read completed, cost %.2fs" % (time.time() - timer))
	return phraseDict
	
# Prefix Tree's default actions should work
def convertToPrefixTree(dict, root=None):
	timer  = time.time()
	if(root is None):
		root = Node(None, None, None)
	for key in dict:
		# traverse and build the tree
		root.buildNode(key, -1, dict[key])
	#if(CALCULATE_TIME):
	#	writeToLog("Tree built, cost %.2fs" % (time.time() - timer))
		
	# force add the endline character to the dict and remove empty key
	endlineChar = next((x for x in root.children if x.data == "\n"), None)
	if(endlineChar is None):
		writeToLog("Make endline character by default")
		endlineChar = Node("\n","\n", root)
		root.addNode(endlineChar)
	endlineChar.children = []
	filter(lambda x: not re.match("^$", x.nodeKey), root.children)
	if(CALCULATE_TIME):
		writeToLog("ConvertToPrefixTree completed, cost %.2fs" % (time.time() - timer))
	return root
	
# Try converting by phrases as long as possible
def convertByPhrasesFromPrefixTree(source, treeRoot, logFile=None, splitOutput=False, contextTag='default'):
	standInNode = Node(None, None, None)
	timer  = time.time()
	currentIndex = 0
	output = []
	splittedSource = []
	currentNode = None
	while(currentIndex < len(source)):
		# use a standInNode in case whole loop have no lastValidNode somehow
		standInNode.data = source[currentIndex]
		lastValidNode = standInNode
		lastValidIdx = -1
		currentNode = treeRoot.seekNodeByChar(source[currentIndex])
		if(currentNode is None):
			# cannot find word at all, import as-is
			output.append(source[currentIndex])
			if(splitOutput):
				splittedSource.append(source[currentIndex])
			currentIndex += 1
			continue
		nextIndex = currentIndex + 1
		while(currentNode is not None and nextIndex < len(source)):
			if(currentNode.data is not None):
				lastValidNode = currentNode
				lastValidIdx = nextIndex
			# find best word available inside the tree
			currentNode = currentNode.seekNodeByChar(source[nextIndex])
			nextIndex += 1
		
		# once found, add the longest phrase to output
		if(isinstance(lastValidNode.data,str)):
			output.append(lastValidNode.data)
		else:
			output.append(lastValidNode.data.getValue(contextTag))
		if(splitOutput):
			splittedSource.append(source[currentIndex:lastValidIdx if lastValidIdx>0 else currentIndex+1])
		if(logFile is not None):
			if(lastValidNode.data is None):
				# There is node initiated yet no data, checkout
				logFile.write("Node name {} - No Data detected.\n".format(lastValidNode.getTrueWord(treeRoot)[0]))
			# logFile.write("{}-{}: {} -> {}\n".format(currentIndex, nextIndex, lastValidNode.getTrueWord(treeRoot), lastValidNode.data))
		currentIndex = nextIndex
	if(CALCULATE_TIME):
		writeToLog("Conversion completed, cost %.2fs, counted %d phrase" % (time.time() - timer, len(output)))
	return output if not splitOutput else (output, splittedSource)
	
# TODO make things more universal
def convertByPhraseFromDict(source, convertDict, maxPhraseLengthSupported=8, logFile=None, splitOutput=False, contextTag='default'):
	timer  = time.time()
	currentIndex = 0
	output = []
	splittedSource = []
	#loopCounter = 0
	while(currentIndex < len(source)):
		#loopCounter += 1
		currentEnd = min(currentIndex+maxPhraseLengthSupported, len(source))
		while(currentEnd > currentIndex):
			searchWord = source[currentIndex:currentEnd]
			if(searchWord in convertDict):
				# found word in dict
				splittedSource.append(searchWord)
				data = convertDict[searchWord]
				if(isinstance(data,str)):
					output.append(data)
				elif(isinstance(data, TranslateData)):
					output.append(data.getValue(contextTag))
				else:
					output.append("$Error$")
				break
			else:
				currentEnd -= 1
		if(currentEnd == currentIndex):
			# Cannot find word in dictionary, import as-is
			splittedSource.append(source[currentIndex])
			output.append(source[currentIndex])
			if(logFile is not None):
				logFile.write("Not found data for word {} in dict.\n".format(source[currentIndex]))
			currentIndex += 1
		else:
			currentIndex = currentEnd
		#if(loopCounter % 200 == 0):
		#	writeToLog("Ran {} loops, currentIdx {}".format(loopCounter, currentIndex))
	if(CALCULATE_TIME):
		writeToLog("Conversion completed, cost %.2fs, counted %d phrase" % (time.time() - timer, len(output)))
	return output if not splitOutput else (output, splittedSource)
	
def convertByPhrasePriorityMode(source, convertDict, maxPhraseLengthSupported=8, logFile=None, splitOutput=False, contextTag='default'):
	timer  = time.time()
	
	currentIndex = 0
	output=[]
	splittedSource=[]
	
	if(isinstance(convertDict, tuple)):
		mergeDict = {}
		mergeDict.update(convertDict[0])
		mergeDict.update(convertDict[1])
		convertDict = mergeDict
	
	# convertDict['\n'] = TranslateData('\n', ("\n\n", 'default'))
	# build list of dict:
	dictByLen = []
	for i in range(maxPhraseLengthSupported):
		dictByLen.append({k: v for k, v in convertDict.items() if len(k) == i+1})
	
	# find phrases with prority: longer is better
	# TODO true word priority
	listTrans = []
	for i in reversed(range(maxPhraseLengthSupported)):
		# Phrases are recorded by a tuple (pos, len)
		idx = 0
		wordLen = i + 1
		newWords = []
		while(idx + wordLen < len(source)):
			word = source[idx:idx+wordLen]
			if(word in dictByLen[i]):
				placeInArray = binarySearch(listTrans, (idx, wordLen), lambda x,y: x[0]>y[0])
				frontWord = listTrans[placeInArray] if placeInArray >= 0 else None
				backWord = listTrans[placeInArray+1] if placeInArray < len(listTrans) and placeInArray >= 0 else None
				if((frontWord is None or frontWord[0] + frontWord[1] <= idx) and (backWord is None or idx + wordLen <= backWord[0])):
					# No word found that take on this word
					newWords.append((idx, wordLen))
					idx += wordLen
				else:
					idx += 1
			else:
				idx += 1
		listTrans.extend(newWords)
		listTrans.sort()
	
	# Translate by the listTrans array
	lastIdx = 0
	for idx, wordLen in listTrans:
		if(lastIdx < idx):
			# Untranslatable phrase left, import as-is
			output.append(source[lastIdx:idx].replace('\n','\n\n'))
			splittedSource.append(source[lastIdx:idx])
		word = source[idx:idx+wordLen]
		data = dictByLen[wordLen-1][word]
		if(isinstance(data, TranslateData)):
			output.append(data.getValue())
		elif(isinstance(data, str)):
			output.append(data)
		else:
			output.append('$NODATA$')
		splittedSource.append(source[idx:idx+wordLen])
		lastIdx = idx+wordLen
	
	if(CALCULATE_TIME):
		writeToLog("Conversion completed, cost %.2fs, counted %d phrase" % (time.time() - timer, len(output)))
	return output if not splitOutput else (output, splittedSource)
	
def convertByPhraseWithHandRule(source, phraseAndNameDict, ruleSet=set({}), separator="", maxPhraseLengthSupported=8, **kwargs):
	"""Convert the data (source) using dictionary (phraseAndNameDict). supporting options ruleSet, separator, maxPhraseLengthSupported"""
	timer = time.time()
	# Will do construct similar to splittedSource
	phraseDict, nameDict = phraseAndNameDict
	dictByLen = []
	for i in range(maxPhraseLengthSupported):
		nextCustomDict = {k: (v, '$PHRASE$') for k, v in phraseDict.items() if len(k) == i+1}
		nextCustomDict.update({k: (v, '$NAME$') for k, v in nameDict.items() if len(k) == i+1})
		dictByLen.append(nextCustomDict)
	
	listTrans = []
	# apply all rules with type before_dict_tran
	beforeRuleDict = [x for x in ruleSet if x.type == 'before_dict_tran']
	# Phrases are recorded by a tuple (pos, len, translation, translationToken)
	idx = 0
	while(idx < len(source)):
		for wordLen in reversed(range(min(maxPhraseLengthSupported, len(source) - idx))):
			# Check by wordLen going from largest to smallest
			wordLen += 1
			word = source[idx:idx+wordLen]
			ruleFound = False
			for rule in beforeRuleDict:
				check = rule.handleRuleOnData(word, separator)
				if(check is not None):
					ruleFound = True
					translation, token = check
					listTrans.append((idx, wordLen, translation, token))
					idx = idx + wordLen
					break
			if(ruleFound):
				# Found rule for the largest string, take and move
				break
		if(not ruleFound):
			# Check again by the next idx if there was no rule worked
			idx += 1
	# print(listTrans)
	
	# find phrases with prority: longer is better
	# TODO true word priority
	for i in reversed(range(maxPhraseLengthSupported)):
		# Phrases are recorded by a tuple (pos, len)
		idx = 0
		wordLen = i + 1
		newWords = []
		while(idx + wordLen < len(source)):
			word = source[idx:idx+wordLen]
			if(word in dictByLen[i]):
				# Check if word found overlap with a detected word before
				placeInArray = binarySearch(listTrans, (idx, wordLen), lambda x,y: x[0]>y[0])
				frontWord = listTrans[placeInArray] if placeInArray >= 0 else None
				backWord = listTrans[placeInArray+1] if 0 < placeInArray+1 < len(listTrans) else None
				if((frontWord is None or frontWord[0] + frontWord[1] <= idx) and (backWord is None or idx + wordLen <= backWord[0])):
					# No overlap found, inserting
					translation, token = dictByLen[i][word]
					if(isinstance(translation, TranslateData)):
						translation = translation.getValue()
					elif(not isinstance(translation, str)):
						translation = '$NODATA$'
						token = '$NODATA$'
					newWords.append((idx, wordLen, translation, token))
					idx += wordLen
				else:
					idx += 1
			else:
				idx += 1
		listTrans.extend(newWords)
		listTrans.sort()
	
	# Apply the listTrans array to create tuple (source, target, token)
	lastIdx = 0
	output = []
	for idx, wordLen, translation, token in listTrans:
		if(lastIdx < idx):
			# Untranslatable phrase left, import as-is; also widen spacing when found
			word = source[lastIdx:idx]
			output.append((word, word.replace('\n','\n\n'), '$UNTRANSLATED$'))
		word = source[idx:idx+wordLen]
		output.append((word, translation, token))
		lastIdx = idx+wordLen
	
	# Run the reordering with after_dict_tran
	afterRuleDict = [x for x in ruleSet if x.type == 'after_dict_tran']
	# Run rules on index basis
	idx = 0
	while(idx < len(output)):
		foundRule = False
		
		for rule in afterRuleDict:
			if(idx+rule.ruleSize > len(output)):
				continue
			dataIn = output[idx:idx+rule.ruleSize]
			dataOut = rule.handleRuleOnData(dataIn, separator)
			if(dataOut is not None):
				# replace the dataIn with tuple dataOut
				del output[idx:idx+rule.ruleSize]
				output.insert(idx, dataOut)
				foundRule = True
			break
		
		# idx + 1 anyway because if found rule, the entire sequence collapsed into one element
		idx += 1
	
	if(CALCULATE_TIME):
		writeToLog("Conversion completed, cost %.2fs, counted %d phrase" % (time.time() - timer, len(output)))
	return output

	
def printTreeToFile(node, rootNode, file):
	# print tree to designated file
	if(isinstance(file, str)):
		file = io.open(file, "w", encoding="utf-8")
	if(node.data is not None):
		if(node.data == "\n"):
			# make an exception for endline char
			file.write("EOL=EOL(default)\n")
		else:
			if(isinstance(node.data, str)):
				file.write("{}={}\n".format(node.getTrueWord(rootNode)[0], node.data))
			elif(isinstance(node.data, TranslateData)):
				file.write(node.data.makeString())
	for child in node.children:
		printTreeToFile(child, rootNode, file)
		
def binarySearch(array, value, compareFunc=lambda x,y: x>y):
	# Check validity
	if(len(array) <= 0):
		writeToLog("Empty array detected @binarySearch")
		return -1
	lowerBound = 0
	upperBound = len(array)-1
	if(compareFunc(array[lowerBound], value) or compareFunc(value, array[upperBound])):
		# writeToLog("Value not in array, {} : {}->{}".format(value, array[upperBound], array[lowerBound]))
		return -1
	elif(array[upperBound] == value):
		return upperBound
	elif(array[lowerBound] == value):
		return lowerBound
	
	while(lowerBound != upperBound):
		if(lowerBound+1 == upperBound):
			# if two last entries are left -> the first one is the phrase index
			# writeToLog("Found? @binarySearch, {} in {}({})-{}({})".format(value, lowerBound, array[lowerBound], upperBound, array[upperBound]))
			return lowerBound
		middle = (lowerBound + upperBound) // 2
		if(array[middle] == value):
			return middle
		elif(compareFunc(array[middle], value)):
			upperBound = middle
		else:
			lowerBound = middle
	# if lowerBound already is upperBound
	return lowerBound
	
def modifyDict(dictionary, source, output, useDataClass=True, replace=False):
	oldVal = dictionary.get(source, None)
	if(oldVal == None or replace):
		# no value/ replace mode, save as default type
		if(useDataClass):
			dictionary[source] = TranslateData(source, [output])
		else:
			dictionary[source] = output
	elif(isinstance(oldVal,str)):
		dictionary[source] = oldVal + '/' + output
	elif(isinstance(oldVal, TranslateData)):
		dictionary[source].addValue(output, tag)
	
def packDictionaryToFile(dict, file):
	if(isinstance(file, str)):
		file = io.open( file, "wb" )
	pickle.dump(dict, file)
	file.close()
	
def getDictionaryFromFile(file):
	if(isinstance(file, str)):
		file = io.open( file, "rb" )
	dict = pickle.load(file)
	file.close()
	return dict
	
class TranslateData:
	listTag = ["default","formal","casual","hostile","other"]
	
	inclusive = 1
	no_relation = 0
	exclusive = -1
	
	VALUE = 0
	TAG = 1
	COND = 2
	
	def __init__(self, name, setValues):
		self.name = name
		self.values = []
		
		for data in setValues:
			if(data is tuple or data is list):
				val = (data[0], data[1], {})
				self.values.append(val)
				if(len(data) > 2):
					for grp in data:
						val[2].get(grp[1], []).append(grp[0])
			elif(isinstance(data, str)):
				val = (data, "default", {})
				self.values.append(val)
			else:
				writeToLog("Cannot read a value at name {}".format(name))
	
	def getValue(self, tag='default', relatedWords=None):
		getFitValues = [v for v in self.values if v[TranslateData.TAG] == tag]
		if(relatedWords is None and len(getFitValues) > 0):
			return getFitValues[0][TranslateData.VALUE]
		elif(relatedWords is not None):
			for val in getFitValues:
				
				for word in relatedWords:
					if(word in val[TranslateData.COND][TranslateData.inclusive]):
						return val[TranslateData.VALUE]
					elif(word in val[TranslateData.COND][TranslateData.exclusive]):
						break
					# else: nothing, continue next word
					
			writeToLog("Search not found with name {} tag {}, use first value.".format(self.name, tag))
		else:
			writeToLog("Node {} have no data, please check again.".format(self.name))
			return "{ERROR}"
	
	def getAllValues(self):
		return [v[0] for v in self.values]
	
	def addValue(self, val, tag='default'):
		valIsIn = [v for v in self.values if v[TranslateData.VALUE] == val and v[TranslateData.TAG] == tag]
		if(valIsIn):
			writeToLog("Value already added before.")
		else:
			self.values.append((val, tag, {}))
	
	def addConditions(self, correctWordVal, type='no_relation', words=[]):
		if(isinstance(type, str)):
			try:
				type = TranslateData.__getattribute__(type)
			except AttributeError:
				writeToLog("Failed to get attribute {} in TranslateData.".format(type))
		_, _, valCond = first(v for v in self.values if v[TranslateData.VALUE] == correctWordVal)
		for word in words:
			if(word in valCond[type]):
				continue
			elif(type != TranslateData.no_relation and word in valCond[-type]):
				# remove word from the type and add it to no_relation
				valCond[-type].remove(word)
				valCond[TranslateData.no_relation].append(word)
	
	def makeString(self):
		firstLine = True
		fullString = ""
		for val in self.values:
			if(firstLine):
				w = self.name + ":\t"
				firstLine = False
			else:
				w = "\t"
			w += "[{}]-tag:{}\n".format(val[0],val[1])
			fullString += w
			
		return fullString
		
class HandRule:
	RULE_TYPE = ['before_dict_tran','after_dict_tran']
	RULE_TOKEN = ['$PHRASE$','$NUMBER$','$NAME$','$PUNCTUATION$','$UNTRANSLATED$']

	def __init__(self, ruleType, ruleRegex, replaceFunction, ruleToken='$PHRASE$'):
		assert ruleType in HandRule.RULE_TYPE, "Invalid ruleType: {}".format(ruleType)
		self.type = ruleType
		self.regex = ruleRegex
		self.replaceFunction = replaceFunction
		self.token = ruleToken

	def handleRuleOnData(self, word, separator=" "):
		if(re.match(self.regex, word)):
			return self.replaceFunction(word), self.token
		else:
			return None

# more pythonic implementation of the HandRule
chineseNumberCharacters = "〇一二三四五六七八九十百千万亿"
chineseDigits = "〇一二三四五六七八九"
chineseScalings = "十百千万亿"
referDict = {k: v for k, v in zip(chineseNumberCharacters, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000000])}
# the regex is for proper chinese number. digits must be followed by a scaling except if it is the last number
# have the end char $ to match whole word
chineseNumberRegex = re.compile("[{1}]*([{0}][{1}]+)+[{0}]*$".format(chineseDigits, chineseScalings))
def numberConvertFunction(numberPhrase):
	# chinese number conversion, assume that the detector already given the phrase
	value = digit = 0
	for idx, char in enumerate(numberPhrase):
		if(char in chineseDigits):
			# dump the value of the previous digit into value
			value += digit
			# normal number, may have following number
			digit = referDict[char]
		elif(char in chineseScalings):
			# scale the value of digit if true
			# if value = 0, the scaling is first, thus imply 1
			if(value == 0):
				digit = 1
			digit *= referDict[char]
		else:
			raise ValueError("Char is invalid, recheck the regex")
	value += digit
	return value
# this rule catch all number in proper format
numberRuleProper = HandRule('before_dict_tran', chineseNumberRegex, numberConvertFunction, '$NUMBER$')

chineseFaultyNumberRegex = re.compile("[%s]{, 3}" % chineseDigits)
stringReferDict = {char: idx for idx, char in enumerate(chineseDigits)}
chineseFaultyConvertFunction = lambda numberString: "".join([stringReferDict[char] for char in numberString])
# this rule catch number in improper format, e.g years
numberRuleBackup = HandRule('before_dict_tran', chineseFaultyNumberRegex, chineseFaultyConvertFunction, '$NUMBER$')
