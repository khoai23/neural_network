import tensorflow as tf
import sys, re, argparse, io, time, pickle
# from itertools import cycle
#from ffBuilder import *
#from exampleBuilder import *
import exampleBuilder, ffBuilder
import numpy as np
from calculatebleu import BLEU as score_bleu
from collections import OrderedDict
from terminal import createTerminal

far_left_token, close_left_token, close_right_token, far_right_token = ['<fl>', '<cl>', '<cr>', '<fr>']
start_token, end_token = ['<s>', '<\s>']
capitalize_token = '<cap>'

conllRegex = re.compile("(\d+)\t(.+)\t_\t([\w$\.,:\"\'-]+)\t([\w$\.,:\"\'-]+)\t_\t(\d+)\t(\w+)")
WORD_WINDOW = 2
def tagChangeFunc(tag):
	return tag + '_<CH>'

def checkCapitalize(word):
	return word[0].isupper() and (len(word) == 1 or not word.isupper())

def parseSentencesFromLines(lines, appendToken=False, lowercase=False, decapitalize=False):
	if(not lowercase):
		sentences = (line.strip().split(' ') for line in lines if(line.strip() != ""))
	else:
		sentences = (line.strip().lower().split(' ') for line in lines if(line.strip() != ""))
	if(decapitalize):
		capFunc = lambda word: [capitalize_token, word.lower()] if checkCapitalize(word) else [word]
		convertSentenceFunc = lambda sentence: [item for word in sentence for item in capFunc(word)]
		sentences = (convertSentenceFunc(sentence) for sentence in sentences)
	if(appendToken):
		startToken, endToken, window = appendToken
		startToken = [startToken] * window
		endToken = [endToken] * window
		modifySentence = lambda sentence: startToken + sentence + endToken
		return (modifySentence(sentence) for sentence in sentences)
	else:
		return sentences

def createBlockMatch(lineOne, lineTwo, debugPrinter=print, ignoreOrder=False):
	# match the sentences as much as possible, and output the respective block
	listMatchIdx = [(0,0), (len(lineOne), len(lineTwo))]
	preferedAfter = 0
	for srcIdx, word in enumerate(lineOne):
		tgtIdx = lineTwo[preferedAfter:].find(word)
		if(tgtIdx < 0):
			# not found in prefered
			tgtIdx = lineTwo.find(word)
			if(tgtIdx >= 0):
				if(debugPrinter is not None):
					debugPrinter("Verbose: found matching words before preference: word {}(from {}) at {}, before {}".format(word, srcIdx, tgtIdx, preferedAfter))
				if(not ignoreOrder):
					# do not keep this pair
					continue
		if(tgtIdx >= 0):
			listMatchIdx.append((srcIdx, tgtIdx))
			preferedAfter = tgtIdx + 1
	# with the matches, slices the lines to equal blocks
	matchingBlocks = []
	for frontTuple, backTuple in zip(listMatchIdx[1:], listMatchIdx[:-1]):
		frontSrcIdx, frontTgtIdx = frontTuple
		backSrcIdx, backTgtIdx = backTuple
		if(frontSrcIdx - backSrcIdx == 1 or frontTgtIdx - backTgtIdx == 1):
			# no correlating group, skip
			continue
		elif(frontSrcIdx > backSrcIdx or frontTgtIdx > backTgtIdx):
			if(ignoreOrder):
				# ignoreOrder will keep the blocks at the expense of potential mismatch
				if(debugPrinter):
					debugPrinter("Keeping {} {} disordered intersecting pair. Sentences: {} {}".format(frontTuple, backTuple, lineOne, lineTwo))
				frontSrcIdx, backSrcIdx = frontSrcIdx, backSrcIdx if frontSrcIdx < backSrcIdx else backSrcIdx, frontSrcIdx
				frontTgtIdx, backTgtIdx = frontTgtIdx, backTgtIdx if frontTgtIdx < backTgtIdx else backTgtIdx, frontTgtIdx
			else:
				# not ignoreOrder mean discarding the blocks. Since we don't keep the conflict orders anyway, here means something gone really really wrong
				raise Exception("Not supposed to came to this branch, recheck the creation of the matches: {}".format(listMatchIdx))
		# append the matches to the result array
		matchingBlocks.append((lineOne[frontSrcIdx+1:backSrcIdx], lineTwo[backSrcIdx+1:backTgtIdx]))
		
		return matchingBlocks

def defaultGraderFunction(lineOne, lineTwo, threshold=0.1, debugPrinter=None):
	# bleuScore the two lines
	score = score_bleu(lineOne, [lineTwo])
	if(score >= threshold):
		# if passed, pair up blocks within the sentences
		return createBlockMatch(lineOne, lineTwo, debugPrinter=debugPrinter)
	else:
		return None

def contextualParseSentencesFromLines(lines, graderFunc, window=0, lowercase=False, decapitalize=False):
	# graderFunc will score sentences' similarity and if two sentence is close enough, create respective relative word-pairs
	assert callable(graderFunc)
	# only compatible with skipgram, since it generate word-pairs only
	if(lowercase):
		# lines must be an array
		lines = [line.lower() for line in lines]
	sentences = [line.strip().split(' ') for line in lines]
	if(decapitalize):
		capFunc = lambda word: [capitalize_token, word.lower()] if checkCapitalize(word) else [word]
		convertSentenceFunc = lambda sentence: [item for word in sentence for item in capFunc(word)]
		sentences = (convertSentenceFunc(sentence) for sentence in sentences)
	
	# if window = 0, compare everything; else only compare sentences whose lengths do not deviate past window
	if(window > 0):
		sentencesDict = {}
		for sentence in sentences:
			if(len(sentence) not in sentencesDict):
				sentencesDict[len(sentence)] = []
			sentencesDict[len(sentence)].append(sentence)

	# after processing, rate the sentences with each other by the grader function
	matchingBlocks = []
	if(window <= 0):
		for firstIdx, firstSentence in enumerate(sentences):
			for secondSentence in sentences[firstIdx:]:
				# by default, the second sentence weight more due to length
				blocks = graderFunc(firstSentence, secondSentence) if(len(secondSentence) >= len(firstSentence)) else graderFunc(secondSentence, firstSentence)
				if(blocks is not None):
					matchingBlocks.extend(blocks)
	else:
		for sentenceLength in sentencesDict:
			# build suitable compare window
			compareWindow = []
			for dev in range(-window, window):
				compareWindow.extend(sentencesDict.get(sentenceLength + dev, []))
			# loop through the sentences with this particular sentenceLength
			fixedLengthSentences = sentencesDict[sentenceLength]
			for firstSentence in fixedLengthSentences:
				for secondSentence in compareWindow:
					if(firstSentence == secondSentence):
						pass
					if(len(firstSentence) > len(secondSentence)):
						# secondSentence should be the longer one
						firstSentence, secondSentence = secondSentence, firstSentence
					blocks = graderFunc(firstSentence, secondSentence)
					if(blocks is not None):
						matchingBlocks.extend(blocks)
			# remove this printer later or use debugPrinter
			print("Finished evaluating for length {}, size of blocks {}".format(sentenceLength, len(matchingBlocks)))
	return matchingBlocks
	
def getRelationFromParentNode(parentNode, listRelation, lookupDict):
	for child in parentNode.children:
		# relation will include dependency to parent, grandparent, parent, child, dependency to child, relating in that order
		# will create relation in tuple of two, and require reversing after
		relations = []
		try:
			grandparent = parentNode.parent
			if(grandparent):
				relations.append((lookupDict.get(tagChangeFunc(parentNode.dependency), 0), lookupDict.get(grandparent.word, 0)))
				relations.append((lookupDict.get(grandparent.word,0), lookupDict.get(parentNode.word, 0)))
		except AttributeError:
			pass
		relations.append((lookupDict.get(parentNode.word, 0), lookupDict.get(child.word, 0)))
		relations.append((lookupDict.get(child.word, 0), lookupDict.get(tagChangeFunc(child.dependency), 0)))
		# Finally append it into listRelation
		listRelation.extend(relations)

def getCBOWRelationFromParentNode(parentNode, listRelation, lookupDict, useGrandparent=False, range=2, defaultWordIdx=0):
	for child in parentNode.children:
		if(useGrandparent):
			if(parentNode.parent is not None):
				grandparentWord = parentNode.grandparent.word
			else:
				grandparentWord = None
			listRelation.append( ([lookupDict.get(grandparentWord, defaultWordIdx), lookupDict.get(tagChangeFunc(parentNode.dependency), defaultWordIdx),
								   lookupDict.get(child.dependency, defaultWordIdx), lookupDict.get(tagChangeFunc(child.word), defaultWordIdx)			], 
								  lookupDict.get(parentNode.word, defaultWordIdx)) )
		else:
			listRelation.append( ([lookupDict.get(parentNode.word, defaultWordIdx), lookupDict.get(tagChangeFunc(parentNode.dependency), defaultWordIdx),
								   lookupDict.get(child.dependency, defaultWordIdx), lookupDict.get(tagChangeFunc(child.tag), defaultWordIdx)			], 
							lookupDict.get(child.word, defaultWordIdx)))
	
def createFeedData(tree, lookupDict, cbowMode=False, cbowGrandparentMode=False):
	# create a tuple (input,output), two matrix representing feeding data. Congregate as needed later
	allRelation = []
	def relationFunc(node):
		if(cbowMode):
			getCBOWRelationFromParentNode(node, allRelation, lookupDict, cbowGrandparentMode)
		else:
			getRelationFromParentNode(node, allRelation, lookupDict)
	
	trueRoot = tree.children[0] if(tree.tag == 'ROOT' and len(tree.children) > 0) else tree
	
	exampleBuilder.recursiveRun(relationFunc, trueRoot)
	exampleBuilder.deleteTree(tree)
	return allRelation
	
def createFeedDataFromSentence(sentence, lookupDict, cbowMode=False, wordWindow=WORD_WINDOW, unknownWordIdx=0, filterFunc=None):
	feed = []
	if(not cbowMode):
		# Normal, get skip-gram within wordWindow
		for i in range(len(sentence)):
			for x in range(1, wordWindow+1):
				if((i+x) < len(sentence)):
					leftWord, rightWord = sentence[i], sentence[i+x]
					feed.append( (lookupDict.get(leftWord, unknownWordIdx), lookupDict.get(rightWord, unknownWordIdx)) )
	else:
		# Assume that we have padding token (add wordWindow(s) number of token <s> and <\s> at the start and end of the sentence):
		for i in range(wordWindow, len(sentence)-wordWindow):
			context = [lookupDict.get(sentence[i+x], unknownWordIdx) for x in range(-wordWindow, wordWindow+1) if x != 0]
			feed.append((context, lookupDict.get(sentence[i], unknownWordIdx)))
	if(filterFunc):
		feed = [f for f in feed if filterFunc(f)]
	return feed

def createFeedDataFromPairBlock(blockPair, lookupDict, certaintyMode=False, unknownWordIdx=0):
	# in certainty mode, the word pair will come with a float number denoting how sure that they are similar. the smaller the respective blocks, the more certain we are
	srcBlock, tgtBlock = blockPair
	if(certaintyMode):
		blockScore = 1.0 / float(len(srcBlock) + len(tgtBlock))
	
	feed = []
	for srcItem in srcBlock:
		for tgtItem in tgtBlock:
			if(certaintyMode):
				feed.append((lookupDict.get(srcItem, unknownWordIdx), lookupDict.get(tgtItem, unknownWordIdx), blockScore))
			else:
				feed.append((lookupDict.get(srcItem, unknownWordIdx), lookupDict.get(tgtItem, unknownWordIdx)))
	return feed

def createExtendedFeedDataFromSentence(sentence, lookupDict, cbowMode=False, wordWindow=WORD_WINDOW):
	# do not use cbow since extended is locked in skipgram - 1
	feed = []
	fl, cl, cr, fr = lookupDict[far_left_token], lookupDict[close_left_token], lookupDict[close_right_token], lookupDict[far_right_token]
	for i in range(len(sentence)):
		for x in range(1, wordWindow+1):
			if((i+x) < len(sentence)):
				leftWord, rightWord = sentence[i], sentence[i+x]
				leftId, rightId = lookupDict.get(leftWord, 0), lookupDict.get(rightWord, 0)
				disLeft, disRight = (fl, fr) if x > 1 else (cl, cr)
				# Our model try to predict the words coming before-after directly
				feed.append(([leftId, disRight], rightId))
				feed.append(([rightId, disLeft], leftId))
	return feed

def _processBatch(batch, cbowMode):
	inputFeed, outputFeed = zip(*batch)
	inputFeed = np.array(inputFeed)
	outputFeed = np.array(outputFeed)
	if(not cbowMode):
		assert len(inputFeed.shape) == 1, "Wrong shape!! {}".format(inputFeed.shape)
		inputFeed = np.expand_dims(inputFeed, -1)
#	print("Batch created with shape {} - {}".format(inputFeed.shape, outputFeed.shape))
	return (inputFeed, outputFeed)

def batchGeneratorFromFeedData(feedGenerator, batchSize, cbowMode=False):
	feed = []
	try:
		for block in feedGenerator:
			feed.extend(block)
			while(len(feed) >= batchSize):
				batch = feed[:batchSize]
				feed = feed[batchSize:]
				yield _processBatch(batch, cbowMode)
				del batch
	except StopIteration:
		if(len(feed) > 0):
			yield _processBatch(feed, cbowMode)
			del feed

def getBatchFromFeedData(batchSize, feed, addRelationFunc, cbowMode=False):
	# Only get a batch of specified size, add more relation 
	raise NotImplementedError("Overhauling the process")
	
def generateDictionaryFromParser(lines, regex, regexGroupIdx, useMatch=True):
	wordDict = {}
	for line in lines:
		match = re.match(regex, line) if(useMatch) else re.search(regex, line)
		if(match is not None):
			if(useMatch):
				word = match.group(regexGroupIdx)
				wordDict[word] = wordDict.get(word, 0) + 1
				# print(word)
			else:
				for arr in match:
					word = arr[regexGroupIdx]
					wordDict[word] = wordDict.get(word, 0) + 1
	
	return wordDict
	
def generateDictionaryFromLines(lines, ignoreDict=None, lowercase=False, decapitalize=False):
	# from a text file, generate the word dict with occurence as value
	# if ignoreDict, only count those not within the ignoreDict
	wordDict = {}
	for line in lines:
		if(lowercase):
			line = line.lower()
		words = line.strip().split()
		for word in words:
			if(decapitalize and checkCapitalize(word)):
				word = word.lower()
				wordDict[capitalize_token] = wordDict.get(capitalize_token, 0) + 1
			if(ignoreDict and word in ignoreDict):
				# if is here, the word is lowercased
				continue
			wordDict[word] = wordDict.get(word, 0) + 1
	return wordDict

def generateDictionaryFromExportedFile(lines):
	# from exported text file, generate the made word dict
	if(len(lines[0].strip().split()) == 2):
		# discard the first line
		lines = lines[1:]
	wordDict = {}
	wordVectors = []
	for line in lines:
		assert len(wordDict) == len(wordVectors)
		word, textVector = line.strip().split("\t ", 1)
		wordDict[word] = len(wordDict)
		wordVectors.append(np.fromstring(textVector))
	wordVectors = np.asarray(wordVectors)
	return wordDict, wordVectors

def organizeDict(wordCountDict, dictSize, extraWords=["*UNKNOWN*"], tagDict=None, existedIndex=0):
	# Dict will be sorted from highest to lowest appearance
	if(dictSize < 0):
		# Obscure mode, remove all items that show up less than |threshold|
		threshold = abs(dictSize)
		wordCountDict = {k: v for k, v in wordCountDict.items() if v > threshold}
		print("Found {:d} words that fit the criteria: appeared more than {:d} times".format(len(wordCountDict), threshold))
	listWords = [w for w,c in sorted(wordCountDict.items(), key=lambda item: item[1], reverse=True)]
	listWords = listWords[:dictSize - len(extraWords)] if(dictSize > 0) else listWords
	# this is important, because we are keeping the unknown words at index 0
	# not as important as it was since we now have the reference to it
	listWords = extraWords + listWords
	if(tagDict):
		for key in tagDict:
			listWords.append(tagChangeFunc(key))
	# create two dictionary for reference
	wordDict = {}
	# the existedIndex is for when there is a dict already occupying that range
	for i, word in enumerate(listWords):
		wordDict[word] = i + existedIndex
	
	del listWords
	return len(wordDict), wordDict

def createEmbedding(parseMode, embMode, fileIn, sizeTuple, extraWords=[], tagDict=None, properCBOW=True, lowercase=False, decapitalize=False):
	dictSize, windowSize, embeddingSize, batchSize = sizeTuple
	# Destroy any current graph and session
	tf.reset_default_graph()
	session = tf.get_default_session()
	if(session is not None):
		session.close()
	
	fileIn.seek(0)
	# Generate frequency dictionary, depending on the input file is a normal one (normal/extended) vs conll one (dependency)
	if(parseMode == 'normal' or parseMode == 'extended'):
		allWordDict = generateDictionaryFromLines(fileIn, lowercase=lowercase, decapitalize=decapitalize)
		fullDictSize = len(allWordDict)
		dictSize, wordDict = organizeDict(allWordDict, dictSize, extraWords=extraWords)
		del allWordDict
	elif(parseMode == 'dependency'):
		allWordDict = generateDictionaryFromParser(fileIn, conllRegex, 2)
		fullDictSize = len(allWordDict)
		dictSize, wordDict = organizeDict(allWordDict, dictSize, extraWords=extraWords, tagDict=tagDict)
		del allWordDict
	elif(parseMode == 'enlarge'):
		assert len(fileIn) == 2, "The input for createEmbedding mode enlarge must be a tuple of (original, improvement)"
		originalFile, enlargeFile = fileIn
		originalWordDict, originalWordVectors = generateDictionaryFromExportedFile(originalFile)
		newWordDict = generateDictionaryFromLines(enlargeFile, ignoreDict=originalWordDict)
		dictSize, newWordDict = organizeDict(newWordDict, dictSize, extraWords=[], existedIndex=len(originalWordDict))
		# merge the old and new word
		wordDict = {**originalWordDict, **newWordDict}
		# hack - keep the original value at None to transfer to enlargement 
		wordDict[None] = len(originalWordDict)
		fullDictSize = len(wordDict)
	else:
		raise ValueError("parseMode argument not supported - {}".format(parseMode))
	refDict = {v:k for k,v in wordDict.items() if k is not None}
	print("All words found: {:d}, effective dictionary size: {:d}".format(fullDictSize, dictSize))
	
	# Create a session based on the actual dictSize (plus unknownWord and tags and maybe start/stop sentence token)
	# Session will only accept this batch size from then on
	# No longer the case, batch is dynamic
	if(parseMode == 'extended'):
		embeddingInputSize = 2
	elif(embMode == 'cbow'):
		# input the size of window
		embeddingInputSize = wordWindow * 2
	else:
		embeddingInputSize = 1
	# extended must use cbow scheme to support its closeleft-left-closeright-right tokens despite being skipgram
	# in contrast, contextual can only use skipgram since there is no way to reflect n-n group relation
	if(parseMode == 'enlarge'):
		sessionTuple = ffBuilder.createEnlargeEmbeddingSession(originalWordVectors, dictSize, embeddingSize, isCBOW=(parseMode == 'cbow'))
	elif((parseMode == 'extended' or (embMode == 'cbow' and parseMode != 'contextual')) and properCBOW):
		sessionTuple = ffBuilder.createEmbeddingSessionCBOW(dictSize, embeddingSize, embeddingInputSize)
	else:
		sessionTuple = ffBuilder.createEmbeddingSession(dictSize, embeddingSize, embeddingInputSize)
	
	dictTuple = dictSize, wordDict, refDict, tagDict
	
	session = sessionTuple[0]
	session.run(tf.global_variables_initializer())
	# print(np.array2string(sessionTuple[4][0].eval(session=session)))
	
	return sessionTuple, dictTuple

def generateTrainingData(parseMode, embMode, fileIn, wordDict, batchSize, cbowGrandparent=False, lowercase=False, decapitalize=False, contextualWindow=0, contextualThreshold=0.1, generatorMode=False):
	"""The function to generate the training data for training
		Args:
			parseMode: the mode to parse the data feed. str
			embMode: the embedding mode to use. str
			wordDict: the dictionary of words. list
			batchSize: the size of the batch to train
			cbowGrandparent: only for dependency+cbow; if true will train the node on its grandparent as well. bool
			lowercase: parse option. lowercase everything. bool
			decapitalize: parse option. all capitalized words will turn into <cap> word. bool
			contextualWindow: only for contextual, specify the window to catch context
			contextualThreshold: only for contextual, specify the similairity for contextual to match
			generatorMode: if true, output a function to create a generators; if false, return a list of batches
		Returns:
			function/list depending on generatorMode
	"""
	fileIn.seek(0)
	cbowMode = (embMode == 'cbow')
	if(parseMode == 'dependency'):
		dataBlock = exampleBuilder.getDataBlock(fileIn.readlines(), exampleBuilder.blankLineRegex)
		dataBlock.pop(0)
		
	# generator function from this block to avoid memory overflow
		def getFeed():
			while len(dataBlock > 0):
				block = dataBlock.pop()
				tree = exampleBuilder.constructTreeFromBlock(block, conllRegex)
				del block
				yield createFeedData(tree, wordDict, cbowMode=cbowMode, cbowGrandparentMode=cbowGrandparent)
	elif(parseMode == 'normal' or parseMode == 'extended'):
		sentenceToDataFunc = createFeedDataFromSentence if parseMode == 'normal' else createExtendedFeedDataFromSentence
		def createFeed():
			fileIn.seek(0)
			sentences = parseSentencesFromLines(fileIn.readlines(), (start_token, end_token, wordWindow) if cbowMode else False, lowercase, decapitalize=decapitalize)
			return (sentenceToDataFunc(sentence, wordDict, cbowMode, wordWindow) for sentence in sentences)
#			return createFeedDataFromSentence(sentence, wordDict, cbowMode, wordWindow)
	elif(parseMode == 'contextual'):
		graderFunc = lambda i1, i2: defaultGraderFunction(i1, i2, threshold=contextualThreshold, debugPrinter=print)
		def createFeed():
			matchingBlocks = iter( contextualParseSentencesFromLines(fileIn.readlines(), graderFunc, lowercase=lowercase, decapitalize=decapitalize, window=contextualWindow) )
			return ( createFeedDataFromPairBlock(block, wordDict) for block in matchingBlocks)
	elif(parseMode == 'enlarge'):
		# filter out those with inputs from entirely original file
		# get the original values from the hack, and check the feed with it
		originalWordSize = wordDict.pop(None)
		checkCaseIsEnlargeCBOW = lambda item: any( (idx >= originalWordSize for idx in item[0]) )
		checkCaseIsEnlargeSkipgram = lambda item: item[0] >= originalWordSize
		# depend on cbow/skipgram
		checkCaseIsEnlarge = checkCaseIsEnlargeCBOW if cbowMode else checkCaseIsEnlargeSkipgram
		def createFeed():
			fileIn.seek(0)
			sentences = parseSentencesFromLines(fileIn.readlines(), (start_token, end_token, wordWindow) if cbowMode else False, lowercase, decapitalize=decapitalize)
			return (createFeedDataFromSentence(sentence, wordDict, cbowMode, wordWindow, filterFunc=checkCaseIsEnlarge) for sentence in sentences)
	else:
		# if we got here, somebody (me) put a new mode in the modeStringParse without actually implementing it. Dumbass.
		raise Exception("Invalid parseMode: {:s}".format(parseMode))

	# if extended, force the cbowMode variable since it is formatted as cbow and we don't want the mirror to happen
	# likewise, if contextual, make sure the data is not parsed in cbowMode either
	cbowMode = (cbowMode or parseMode == 'extended') and parseMode != 'contextual'
	# initialize
	if(not generatorMode):
		feed = getFeed(reset=True)
		batch = getBatchFromFeedData(batchSize, feed, getFeed, cbowMode)
		# loop until no more feed/batch can be generated
		workingBatches = []
		while(batch is not None):
			workingBatches.append(batch)
			batch = getBatchFromFeedData(batchSize, feed, getFeed, cbowMode)
		
		return workingBatches
	else:
		def createBatchGeneratorFunc():
			feedGenerator = createFeed()
			return batchGeneratorFromFeedData(feedGenerator, batchSize, cbowMode=cbowMode)
		return createBatchGeneratorFunc

def trainEmbedding(createdData, sessionTuple, epoch, timerFunc=None, timerInterval=1000, passedSteps=0):
	session, train_op, training_inputs, training_outputs, resultTuple = sessionTuple
	if(callable(createdData)):
		print("Generator function received @trainEmbedding")
		createdDataFn = createdData
	else:
		createdDataFn = None
	for i in range(1, epoch+1):
		if(createdDataFn):
			createdData = createdDataFn()
			print("Re-initiate data for a new cycle: {}".format(createdData))
#			assert isinstance(createdData, types.GeneratorType)
		total_loss = per_timer_loss = 0.0
		counter = 0
		for inputBatch, outputBatch in createdData:
			_, loss = session.run(fetches=train_op, feed_dict={training_inputs:inputBatch, training_outputs:outputBatch})
			passedSteps += 1
			counter += 1
			per_timer_loss += loss
			if(passedSteps % timerInterval == 0 and timerFunc is not None):
				total_loss += per_timer_loss
				print("Steps {:d}; time passed {:.2f}s, average loss {:.4f}.".format(passedSteps, timerFunc(None), per_timer_loss / float(timerInterval)))
				per_timer_loss = 0
		currentTime = timerFunc(i)
		print("Epoch {:d} completed, time passed {:.2f}s, total loss in total / per batch: {:.5f} / {:.5f}".format(i, currentTime, total_loss, total_loss / float(counter) * timerInterval))
	return passedSteps

def evaluateSimilarity(divResult):
	return 1 - np.tanh(np.abs(np.log(divResult)))

def findSimilarity(tupleMatrix, refDict, sample, mode=3, checkSize=10):
	embeddingMatrix, normalizedMatrix = tupleMatrix
	sampleMatrix = np.transpose([normalizedMatrix[i] for i in sample])
	sampleSize = len(sample)
	# print([[i for i in vector] for vector in sampleMatrix])
	# print(embeddingMatrix.shape, vectorLength.shape)
	# calculate distance for normalized version
	if(mode == 2 or mode == 3):
		distance = np.transpose(np.matmul(normalizedMatrix, sampleMatrix))
		print("Check normalized version:")
		for idx in range(sampleSize):
			word = refDict[sample[idx]]
			nearest = (-distance[idx, :])
			# print(nearest)
			nearest = nearest.argsort()[:checkSize + 1]
			nearest = [refDict[x] for x in nearest]
			print("{} -> {}".format(word, nearest))
	
	if(mode == 1 or mode == 3):
		vectorLength = np.linalg.norm(embeddingMatrix, axis=1)
		magDifference = np.asarray([vectorLength / vectorLength[sample] for sample in sample])
		magDifference = 1 - np.tanh(np.abs(np.log(magDifference)))
		# multiply the normalized part with the difference of magnitude converted into 0-1 range (1 is the same, 0 is infinitively different)
		print("Check normal version:")
		#print(distance.shape, magDifference.shape)
		distance = np.multiply(magDifference, distance)
		for idx in range(sampleSize):
			word = refDict[sample[idx]]
			nearest = -distance[idx]
			nearest = nearest.argsort()[:checkSize + 1]
			nearest = [refDict[x] for x in nearest]
			print("{} -> {}".format(word, nearest))
	

def evaluateEmbedding(sessionTuple, combinedDict, sampleSize, sampleWordWindow, checkSize, sample=None):
	refDict = combinedDict[2]
	# Taken straight from the basic word2vec
	# use numpy instead of tf.matmul
	session=sessionTuple[0]
	
	# sampleSize samples in range of 0-wordWindow, no duplication
	random_sample = np.random.choice(sampleWordWindow, sampleSize, replace=False) if sample is None else sample
#	const_phr = ['gia_đình','ngành', 'triệu', 'điều']
#	random_sample = [combinedDict[1][phr] for phr in const_phr]
	
	# get the un-normalized version and check for closest values
	_, _, _, _, resultTuple = sessionTuple
	embeddingMatrix = resultTuple[0].eval(session=session)
	normalizedMatrix = resultTuple[1].eval(session=session)
	# print(np.array2string(normalizedMatrix[2:6]))
	findSimilarity((embeddingMatrix, normalizedMatrix), refDict, random_sample, checkSize=checkSize)

	return embeddingMatrix, normalizedMatrix

def findAndPrintNearest(tupleMatrix, combinedDict, wordList, mode=3, checkSize=11):
	wordDict = combinedDict[1]
	refDict = combinedDict[2]
	
	# convert wordList(word) to sample(ids)
	sample = []
	for word in wordList:
		if(word in wordDict):
			sample.append(wordDict[word])
		else:
			print("Word {} not found in dictionary.".format(word))
	if(len(sample) == 0):
		print("No acceptable words, exit @findAndPrintNearest.")
		return
	
	findSimilarity(tupleMatrix, refDict, sample, mode=mode, checkSize=checkSize)

def exportEmbedding(exportMode, outputDir, outputExt, wordDict, resultMatrixTuple, embeddingCountAndSize, tagDictForRemoval=None):
	if(exportMode == "all" or exportMode == "both"):
		# print both the default and the normalized version into files
		file_default = io.open(outputDir + '_default.' + outputExt, 'w', encoding = 'utf-8')
		file_normalized = io.open(outputDir + '_normalized.' + outputExt, 'w', encoding = 'utf-8')
		file_default.write(embeddingCountAndSize)
		file_normalized.write(embeddingCountAndSize)
		for word in wordDict:
			if(tagDictForRemoval and word in tagDictForRemoval):
				continue
			idx = wordDict[word]
			writeWordToFile(file_default, word, resultMatrixTuple[0][idx])
			writeWordToFile(file_normalized, word, resultMatrixTuple[1][idx])
		file_default.close()
		file_normalized.close()
	elif(exportMode == "default" or exportMode == "normalized"):
		useDict = 0 if(exportMode == "default") else 1
		resultMatrix = resultMatrixTuple[useDict]
		with io.open(outputDir + '.' + outputExt, 'w', encoding = 'utf-8') as writeFile:
			writeFile.write(embeddingCountAndSize)
			for word in wordDict:
				if(tagDictForRemoval and word in tagDictForRemoval):
					continue
				idx = wordDict[word]
				writeWordToFile(writeFile, word, resultMatrix[idx])
	elif(exportMode == "binary" or exportMode == "binary_full"):
		# Only export the normal version in normal binary, since the normalized one can be made from it
		# use pickle
		normalDict = OrderedDict()
		if("full" in exportMode):
			normalizedDict = OrderedDict()
		assert len(wordDict) == len(resultMatrixTuple[0]) and len(wordDict) == len(resultMatrixTuple[1])
		for word in wordDict:
			idx = wordDict[word]
			normalDict[word] = resultMatrixTuple[0][idx]
			if("full" in exportMode):
				normalizedDict[word] = resultMatrixTuple[1][idx]
		with io.open(outputDir + '.' + outputExt, "wb") as pickleFile:
			pickle.dump(normalDict if exportMode == "binary" else (normalDict, normalizedDict), pickleFile)
	elif(exportMode == "dict_and_matrix"):
		with io.open(outputDir + '.' + outputExt, "wb") as pickleFile:
			pickle.dump((wordDict, resultMatrixTuple[0]), pickleFile)
	else:
		raise Exception("Wrong mode @exportEmbedding, must be all|both|default|normalized|binary|binary_full|dict_and_matrix")

def writeWordToFile(writeFile, word, vector, dimensionFormat="{:.6f}"):
	writeFile.write(word + '\t' + " ".join((dimensionFormat.format(dim) for dim in vector)) + "\n")

def writeListWordsToFile(fileOrFileDir, wordDict):
	if(isinstance(fileOrFileDir, str)):
		writeFile = io.open(fileOrFileDir, 'w', encoding = 'utf-8')
	else:
		writeFile = fileOrFileDir
	for word in wordDict:
		writeFile.write(word + '\n')
	writeFile.close()
	
def modeStringParse(string):
	# mode is basically (isNormal, isCBOW, windowsize) tuple here
	string = string.lower().split('_')
	if(any(string[0] == item for item in ['normal', 'dependency', 'extended', 'contextual', 'enlarge'])):
		isNormal = (string[0] == 'normal')
	else:
		raise argparse.ArgumentTypeError('Arg1 must be dependency/normal/extended/contextual/enlarge')
	
	if(any(string[1] == item for item in ['skipgram', 'cbow'])):
		isCBOW = (string[1] == 'cbow')
	else:
		raise argparse.ArgumentTypeError('Arg2 must be skipgram/cbow')
	
	try:
		windowSize = int(string[2])
	except Exception as e:
		print("Error getting windowSize from {:s}, default to {:d}".format(e, WORD_WINDOW))
		windowSize = WORD_WINDOW
	return isNormal, isCBOW, windowSize, string
	
def runTerminal(args, items=None):
		data = {}
		data["select"] = 3
		data["neighborNum"] = 10
		if(not args.terminal_commands_only and items is not None):
			resultTuple, dictTuple = items
			data["embedding"] = resultTuple
			data["dict"] = dictTuple
		elif(args.terminal_commands_only):
			print("Warning: in terminal command only mode, you do not have data. You must load a previous dump.")
		else:
			raise ValueError("Items is None while not args.terminal_commands_only.")

		commandDict = {}
		def modeSelector(data):
			try:
				data["select"] = int(input("1 for full, 2 for normalized, 3 for both: "))
				data["neighborNum"] = int(input("Number of neighbors to be shown: "))
			except ValueError as e:
				print("Invalid value. Error: {}".format(e))
			return data
		commandDict["m"] = modeSelector
		commandDict["mode"] = modeSelector
		
		def showNeighbor(data):
			wordOrWords = input("The comparing words: ").strip()
			if(wordOrWords.find(" ") >= 0):
				listWords = wordOrWords.split()
			else:
				listWords = [wordOrWords]
			findAndPrintNearest(data["embedding"], data["dict"], listWords, mode=data["select"], checkSize=data["neighborNum"])
		commandDict["i"] = showNeighbor
		commandDict["input"] = showNeighbor
		
		def saveData(data):
			path = input("Specify the save path: ")
			saveFile = io.open(path, "wb")
			pickle.dump(data, saveFile)
			print("Dumped the needed data @{:s}".format(path))
		commandDict["s"] = saveData
		commandDict["save"] = saveData

		def loadData(data):
			path = input("Specify the load path: ")
			loadFile = io.open(path, "rb")
			data = pickle.load(loadFile)
			print("Loaded the needed data @{:s}".format(path))
			return data
		commandDict["l"] = loadData
		commandDict["load"] = loadData
		
		startLine = "+++ Console Interactions Ready +++"
		createTerminal(data, commandDict, initialDisplayString=startLine)
#		helperLine = "q/quit to exit, m/mode to select comparing embedding, e/export to try export a distribution image, i/input to input a list of words for comparison, s/save and l/load to save or load the command to directory"

if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Perform embedding for words in input file.')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input files')
#	parser.add_argument('-m', '--mode', type=modeStringParse, required=True, help='the mode to embed the word2vec in, must be in format (dependency|normal|extended)_(skipgram|cbow)_wordWindow(only if in normal mode)')
	parser.add_argument('--parse_mode', type=str, choices=["normal", "dependency", "extended", "enlarge"], default="normal", help="Choice of parseing process")
	parser.add_argument('--embedding_mode', type=str, choices=["skipgram", "cbow"], default="skipgram", help="Choice of embedding process")
	parser.add_argument('--embedding_window', type=int, default=2, help="Size of window to train embedding")
	parser.add_argument('-x', '--export_mode', required=True, type=str, help='exporting the values to an outside file, must be (all|both|default|normalized|binary|binary_full|vocab)')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-t','--tagdir', type=str, default="all_tag.txt", help='location of the tag file containing both POStag and dependency, default all_tag.txt')
	parser.add_argument('--input_extension', type=str, default="conllx", help='file extension for input file, default conllx')
	parser.add_argument('--output_extension', type=str, default=None, help='file extension for output file, default embedding.txt/embedding.bin')
	parser.add_argument('--unknown_word', type=str, default="<unk>", help='placeholder name for words not in dictionary, default unk')
	parser.add_argument('--epoch', type=int, default=1000, help='number of iterations through the data, default 1000')
	parser.add_argument('--batch_size', type=int, default=512, help='size of the batches to be feed into the network, default 512')
	parser.add_argument('--embedding_size', type=int, default=100, help='size of the embedding to be created, default 100')
	parser.add_argument('--dict_size', type=int, default=10000, help='size of the words to be embedded, default 10000, input -1 for all words, -n (int) for taking only those occurred more than n times.')
	parser.add_argument('-e','--evaluate', type=int, default=0, help='try to evaluate the validity of the trained embedding. Note that at the end of the training the evaluation function will fire regardless of this value. A positive number for the steps where you launch the evaluation')
	parser.add_argument('--filter_tag', action='store_true', help='remove the trained tag from the output file, dependency mode only')
	parser.add_argument('--grandparent', action='store_true', help='use grandparent scheme, only available to dependency_cbow mode')
	parser.add_argument('--average', action='store_false', help='use average tensor instead of fully independent tensor, only available to normal_cbow mode')
	parser.add_argument('--timer', type=int, default=1000, help='the inteval to call timer func')
	parser.add_argument('--lowercase', action='store_true', help='do the lowercase by the default python function. Not recommended.')
	parser.add_argument('--decapitalize', action='store_true', help='create a <cap> token before capitalized words')
	parser.add_argument('--terminal_commands', action='store_true', help='run a terminal after training to check on the result')
	parser.add_argument('--terminal_commands_only', action='store_true', help='run a terminal only, doing nothing about the script itself')
	parser.add_argument('--context_window', type=int, default=2, help='the maximum difference to run compare sentences. Only used in contextual mode. Default 2.')
	parser.add_argument('--context_threshold', type=float, default=0.1, help='the threshold to consider extracting group. Only used in contextual mode. Default 0.1')
	parser.add_argument('--generator_mode', action='store_true', help='if specified, will set the data to generator mode, which will prevent overflowing.')
	parser.add_argument('--enlarge_source_file', type=str, default=None, help='the original embedding file needing enlargement. Required in enlarge parse mode ')
	parser.add_argument('--other_mode', action='store_true', help='placeholder')
	args = parser.parse_args()
	
	# if in this mode, ignore all that came after
	if(args.terminal_commands_only):
		runTerminal(args)
		sys.exit()
	
	if(args.outputdir is None):
		args.outputdir = args.inputdir
	if(args.output_extension is None):
		if('binary' in args.export_mode):
			args.output_extension = "embedding.bin"
		else:
			args.output_extension = "embedding.txt"
	embeddingSize = args.embedding_size
	batchSize = args.batch_size
	dictSize = args.dict_size
	epoch = args.epoch
	unknownWord = args.unknown_word
	sampleSize = 8
	sampleWordWindow = 200
	checkSize = 10
#	isNormal, isCBOW, wordWindow, modeString = args.mode
	parseMode, embMode, wordWindow = args.parse_mode, args.embedding_mode, args.embedding_window
	
	timer = time.time()
	
	inputFile = io.open(args.inputdir + '.' + args.input_extension, 'r', encoding='utf-8')
	# Exit prematurely with vocab export
	if(args.export_mode == 'vocab'):
		allWordDict = generateDictionaryFromLines(inputFile.readlines(), lowercase=args.lowercase, decapitalize=args.decapitalize)
		writeListWordsToFile(args.outputdir + '.' + args.output_extension, allWordDict)
		print("Done for vocab export, time passed %.2fs" % (time.time() - timer))
		sys.exit(0)
	# Initialize the embedding
	# Todo add the inverse mode
	inputs = inputFile #.readlines()
	tagDict = None
	if(parseMode == 'normal'):
		# add the sos and eos token as well
		extraWords = [unknownWord, start_token, end_token]
	elif(parseMode == 'extended'):
		# add far left - close left - close right - far right
		extraWords = [unknownWord, far_left_token, close_left_token, close_right_token, far_right_token, start_token, end_token]
	elif(parseMode == 'enlarge'):
		# do nothing to extra word, but open the file to inputs and read them as well
		with io.open(args.enlarge_source_file, "w", encoding="utf-8") as sourceFile:
			inputs = (sourceFile.readlines(), inputs)
	elif(parseMode == 'dependency'):
		tagDict = exampleBuilder.getTagFromFile(args.tagdir, True)
		# add only the unknownWord
		extraWords = [unknownWord]
	else:
		raise argparse.ArgumentError("mode", "parseMode not implemented {:s}".format(parseMode))
	sessionTuple, dictTuple = createEmbedding(parseMode, embMode, inputs, (dictSize, wordWindow, embeddingSize, batchSize), tagDict=tagDict, properCBOW=args.average, extraWords=extraWords, lowercase=args.lowercase, decapitalize=args.decapitalize)
#	sessionTuple, dictTuple = createEmbedding((file, isNormal, isCBOW, wordWindow, args.average, args.lowercase), tagDict, dictSize, embeddingSize, batchSize, unknownWord if(not isNormal) else [unknownWord, '<s>', '<\s>'])
	print("Done for @createEmbedding, time passed %.2fs" % (time.time() - timer))
	
	# create the static sample for evaluation
	# make sure it belong to the new dict
	static_sample = np.random.choice(sampleWordWindow, sampleSize, replace=False)
	# Train and evaluate the embedding
	def timerFunc(counter=None):
		if(args.evaluate > 0 and counter is not None):
			if(counter % args.evaluate == 0):
				evaluateEmbedding(sessionTuple, dictTuple, sampleSize, sampleWordWindow, checkSize, sample=static_sample)
		return time.time() - timer
		
	wordDict = dictTuple[1]
	generatedTrainData = generateTrainingData(parseMode, embMode, inputFile, wordDict, batchSize, cbowGrandparent=args.grandparent, lowercase=args.lowercase, contextualWindow=args.context_window, contextualThreshold=args.context_threshold, generatorMode=args.generator_mode)
	print("Done generating training data, time passed {:.2f}s, generated batch of size {:d}".format(time.time() - timer, batchSize))
	if(args.generator_mode):
		print("Is a generator maker func: ", generatedTrainData)
		assert callable(generatedTrainData)
	
	totalSteps = trainEmbedding(generatedTrainData, sessionTuple, epoch, timerFunc=timerFunc, timerInterval=args.timer)
	print("All training complete @trainEmbedding, total steps {:d}, time passed {:.2f}s".format(totalSteps, time.time() - timer))
	inputFile.close()
	
	# Final evaluation. Must run to bring out the resultTuple
	resultTuple = evaluateEmbedding(sessionTuple, dictTuple, sampleSize, sampleWordWindow, checkSize)
	print("Final @evaluateEmbedding on random sample, time passed %.2fs" % (time.time() - timer))
	
	dictSize = dictTuple[0]
	embeddingCountAndSize = "{} {}\n".format(dictSize, embeddingSize)
	
	# Export to file
	exportEmbedding(args.export_mode, args.outputdir, args.output_extension, dictTuple[1], resultTuple, embeddingCountAndSize)

	if(args.terminal_commands):
		# Allow reading words from the terminal and output closest words found
		runTerminal(args, (resultTuple, dictTuple))
