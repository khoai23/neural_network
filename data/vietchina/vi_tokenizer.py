import sys, getopt, re, time

WORDS_FILE = "Viet74K.txt"
TEXT_FILE = "vi_untokenized.txt"
OUTPUT_FILE = "vi_tokenized.txt"
MAX_PHRLEN = 6
PRE_SPLITTED = True
LONG_NAME_CHECK = True
RUN_REPLACEMENT_SCHEME = True

# Replacement is in right/wrong checker
numberRegex = re.compile("([\d\.,\\/\-%]+)")
allCharString = 'AaĂăÂâBbCcDdĐđEeÊêGgHhIiKkLlMmNnOoÔôƠơPpQqRrSsTtUuƯưVvXxYyẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
validWord = re.compile("[{}]+".format(allCharString))
knownSpecialCharacter = re.compile("[,\"\']")
moreThanOneSpace = re.compile("( ){2,}")
try: 
	wordsFile = open(WORDS_FILE, "r", encoding="utf-8")
	textFile = open(TEXT_FILE, "r", encoding="utf-8")
	output = open(OUTPUT_FILE, "w", encoding="utf-8")
except IOError:
	print('Cannot read file, exiting.')
	sys.exit()

wordDict = {}
# Secondary dict are added to check on uppercases word for stressing purpose.
secondaryDict = {}
timer = time.time()
for word in wordsFile.readlines():
	word = word.lower().strip()
	wordDict[word] = 0
	# secondaryDict will contain the correct form
	secondaryDict[word.capitalize()] = word
allUniqueReplacement = [('ể','ể'), ('ụ','ụ'), ('​​ban',)]
if(RUN_REPLACEMENT_SCHEME):
	pass
print("Dict read, cost {}, length {}".format(time.time()-timer, len(wordDict)))

timer = time.time()
text = textFile.read()
endSentenceDemi = re.compile("([\.!?\"\':])")
allPunctuationDemi = re.compile("([\.!?\"\'“”,，;:\(\)])")
if(PRE_SPLITTED):
	# add extra space for these demi, and make sure there is no two-space gap
	replaceDemi = r" \1 "
	text = re.sub(allPunctuationDemi, replaceDemi, text)
	text = re.sub(moreThanOneSpace, ' ', text)
	listSentence = re.split('\n', text)
else:
	allUnusedCharacter = re.compile("[\r​]")
	customizedEndlineChar = '<~eol~>'
	text = re.sub(allUnusedCharacter, '', text).replace('\n','.\n').replace(',', " ,").replace('\n','')
	text = re.sub(moreThanOneSpace, ' ', text).replace('“','\"').replace('”','\"')
	# text = text.replace("\n", "").replace("\r", "").replace(",", " ,").replace("-","")
	text = re.split(endSentenceDemi, text)
	listSentence = list(filter(lambda x: not re.match(r'^\s*$', x), text)) #text.split("\.|\"|;|:")
temp = []
wordNameDict = {}
if(LONG_NAME_CHECK):
	allInvalidWords = {}
	def isPossibleName(word):
		if(not isinstance(word, str)):
			print("Error: input word ", word)
		return len(word) > 0 and word[0].isupper()
	def isPossibleLongName(words):
		# print(words, all(isPossibleName(word) for word in words))
		return all(isPossibleName(word) for word in words)
	def flattenList(l):
		return [item for sublist in l for item in sublist]
	for line in listSentence:
		allWords = line.strip().split()
		for word in allWords:
			if(re.match(validWord, word) is None and re.match(numberRegex, word) is None):
				allInvalidWords[word] = word
		# This version will lose short names if they are starting the sentence
		allPossibleName = [ ' '.join(words) for words in ## join it back together
								flattenList((allWords[x:x+size] for x in range(len(allWords) - size + 1)) ## for all words of size <size> founded in sentence
								for size in range(2, MAX_PHRLEN+1)) ## for all size between 2 and MAX_PHRLEN
							if(isPossibleLongName(words))] ## if they passed the prerequisite
		# Short names not at start of sentences can be recorded, but appear to be a great danger. Detect sos as first words and behind demi 
		# allPossibleName.extend([allWords[idx] for idx in range(1, len(allWords)) if isPossibleName(allWords[idx]) and not re.match(endSentenceDemi, allWords[idx-1]) is None])
		# full uppercase words should be included as well
		allPossibleName.extend([word for word in allWords if(word.isupper())])
		# Trim the list for the longest ones available, feed them in if good enough
		for name in allPossibleName:
			if(all(name not in otherName or name == otherName for otherName in allPossibleName)):
				wordNameDict[name] = wordNameDict.get(name, 0) + 1
else:
	for line in listSentence:
		#output.write(line + "\n")
		words = line.strip().split(" ")
		if(len(words) > 0 and not words[0].isupper()): # first word all in uppercase would be left as-is
			words[0] = words[0].lower()
		temp.append(words)
		currentName = None
		for word in words:
			if(len(word) == 0):
				continue
			if(not word[0].isupper()):
				if(currentName is not None):
					# print("add name {}".format(currentName))
					if(currentName not in secondaryDict):
						# Do not count all single-letter name
						wordNameDict[currentName] = wordNameDict.get(currentName, 0) + 1
					currentName = None
			else:
				if(currentName is None):
					currentName = word
				else:
					currentName += ' ' + word 
print("Name searched, cost {}, wordNameDict len {}".format(time.time()-timer, len(wordNameDict)))

timer = time.time()
# now we have both wordDict and nameDict, try to split to phrases
tokenized = []
for line in listSentence:
	# print("Line being processed: {}".format(line))
	words = line.strip().split(" ")
	idx = 0
	partition = []
	searchFirstWord = ''
	extraIdx = 0
	# First word will check the lowercased version first and revert if not found
	"""firstWord = words[0]
	words[0] = words[0].lower()
	for firstEnd in range(1,MAX_PHRLEN):
		searchFirstWord =' '.join(words[idx:firstEnd])
		if(searchFirstWord in wordDict or searchFirstWord in wordNameDict):
			# compareFirstWord = firstEnd
			# partition.append(searchWord)
			# idx = firstEnd
			extraIdx = firstEnd
			break
		if(firstEnd == 1):
			# No word found, reset the searchFirstWord var
			searchFirstWord = ''
			extraIdx = -1
	#if(idx == 0):
	# Revert for first word check
	words[0] = firstWord"""
	while(idx < len(words)):
		if(re.match(knownSpecialCharacter,words[idx])):
			partition.append(words[idx])
			idx += 1
			continue
		foundWord = False
		eop = min(idx + MAX_PHRLEN, len(words))
		while(eop > idx):
			searchWord = ' '.join(words[idx:eop]) if(eop > idx+1) else words[idx]
			# print(searchWord)
			if(searchWord in wordDict or searchWord in wordNameDict):
				# print(searchWord)
				# found word, repeating
				foundWord = True
				# choose the best length phrase or name instead if first idx
				partition.append(searchWord)
				idx = eop
				break
			elif(searchWord in secondaryDict):
				foundWord = True
				partition.append(secondaryDict[searchWord]) # searchWord.lower()
				idx = eop
			eop -= 1
		"""if(idx == 0):
			# Check word found in uppercase (searchword, eop) and lowercase (searchFirstWord, extraIdx)
			if(not re.match(validWord, words[0])):
				# print("Invalid type, import as-is")
				partition.append(words[0].lower())
				idx += 1
			elif(not(searchWord == '') and extraIdx > 0):
				if(foundWord):
					# Both found, check against each other
					if(len(searchWord) > len(searchFirstWord)):
						# print ("searchWord win, {}".format(eop))
						partition.append(searchWord)
						idx = eop
					else:
						# print ("searchFirstWord win, {}".format(extraIdx))
						partition.append(searchFirstWord)
						idx = extraIdx
				else:
					# Only lowercase phrase found
					# print ("searchFirstWord only, {}".format(extraIdx))
					partition.append(searchFirstWord)
					idx = extraIdx
			else:
				if(foundWord):
					# Only uppercase phrase found
					# print ("searchWord only, {}".format(eop))
					partition.append(searchWord)
					idx = eop
				else:
					# None found.
					# print("None")
					partition.append(words[0].lower())
					idx += 1
		el"""
		if(not foundWord):
			if(re.match(numberRegex, words[idx])):
				# a number construct, split it into numbers
				partition.extend(words[idx])
			else:
				# single word not found in dictionary, import as-is
				partition.append(words[idx])
			idx += 1
	tokenized.append(partition)
	searchWord = None
print("Splitted sentences, cost {}".format(time.time()-timer))

# detect those which are numbers/date


timer = time.time()
# print the data
for line in tokenized:
	output.write(' '.join(line) + '\n')
#output.write("\n\n\nNames found:\n")
#for name in wordNameDict:
#	output.write("[{}]:{}\n".format(name, wordNameDict[name]))
#if(allInvalidWords):
#	output.write("\n\n\Invalid found:\n")
#	for key in allInvalidWords:
#		output.write("[{}]:{}\n".format(key, wordNameDict[name]))
print("Printed to file, cost {}".format(time.time()-timer))
output.close(); sys.exit()