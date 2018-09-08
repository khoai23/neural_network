import tensorflow as tf
import sys, re, argparse, io, time, pickle
from ffBuilder import *
from exampleBuilder import *
import numpy as np
from collections import OrderedDict

far_left_token, close_left_token, close_right_token, far_right_token = ['<fl>', '<cl>', '<cr>', '<fr>']
start_token, end_token = ['<s>', '<\s>']
capitalize_token = '<cap>'

conllRegex = re.compile("(\d+)\t(.+)\t_\t([\w$\.,:\"\'-]+)\t([\w$\.,:\"\'-]+)\t_\t(\d+)\t(\w+)")
WORD_WINDOW = 2
def tagChangeFunc(tag):
	return tag + '_<CH>'

def checkCapitalize(word):
	return word[0].isupper() and (len(word) == 0 or not word.isupper())

def parseSentencesFromLines(lines, appendToken=False, lowercase=False, decapitalize=False):
	if(not lowercase):
		sentences = (line.strip().split(' ') for line in lines if(line.strip() != ""))
	else:
		sentences = (line.strip().lower().split(' ') for line in lines if(line.strip() != ""))
	if(decapitalize):
		capFunc = lambda word: [capitalize_token, word.lower()] if checkCapitalize(word) else [word]
		convertSentenceFunc = lambda sentence: [item for word in sentence for item in capFunc(word)]
		sentence = (convertSentenceFunc(sentence) for sentence in sentences)
	if(appendToken):
		startToken, endToken, window = appendToken
		startToken = [startToken] * window
		endToken = [endToken] * window
		modifySentence = lambda sentence: startToken + sentence + endToken
		return (modifySentence(sentence) for sentence in sentences)
	else:
		return sentences
	
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
								  lookupDict.get(parent.word, defaultWordIdx)) )
		else:
			listRelation.append( ([lookupDict.get(parentNode.word, defaultWordIdx), lookupDict.get(tagChangeFunc(parentNode.dependency), defaultWordIdx),
								   lookupDict.get(child.dependency, defaultWordIdx), lookupDict.get(tagChangeFunc(child.tag), defaultWordIdx)			], 
							lookupDict.get(child.word, defaultWordIdx)))
	
def createFeedData(tree, lookupDict, cbow=(False, False)):
	# create a tuple (input,output), two matrix representing feeding data. Congregate as needed later
	allRelation = []
	cbowMode, cbowGrandparent = cbow
	def relationFunc(node):
		if(cbowMode):
			getCBOWRelationFromParentNode(node, allRelation, lookupDict, cbowGrandparent)
		else:
			getRelationFromParentNode(node, allRelation, lookupDict)
	
	trueRoot = tree.children[0] if(tree.tag == 'ROOT' and len(tree.children) > 0) else tree
	
	recursiveRun(relationFunc, trueRoot)
	deleteTree(tree)
	return allRelation
	
def createFeedDataFromSentence(sentence, lookupDict, cbowMode=False, wordWindow=WORD_WINDOW):
	feed = []
	if(not cbowMode):
		# Normal, get skip-gram within wordWindow
		for i in range(len(sentence)):
			for x in range(1, wordWindow+1):
				if((i+x) < len(sentence)):
					leftWord, rightWord = sentence[i], sentence[i+x]
					feed.append((lookupDict.get(leftWord, 0), lookupDict.get(rightWord, 0)))
	else:
		# Assume that we have padding token (add wordWindow(s) number of token <s> and <\s> at the start and end of the sentence):
		for i in range(wordWindow, len(sentence)-wordWindow):
			context = [lookupDict.get(sentence[i+x], 0) for x in range(-wordWindow, wordWindow+1) if x!=0]
			feed.append((context, lookupDict.get(sentence[i], 0)))
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

def getBatchFromFeedData(batchSize, feed, addRelationFunc, cbowMode=False):
	# Only get a batch of specified size, add more relation 
	while(len(feed) < batchSize):
		newRelations = addRelationFunc()
		if(newRelations is None):
			break
		else:
			feed.extend(newRelations)
			# in skip-gram, double up the relation
			if(not cbowMode):
				reverseRelation = [(w2, w1) for w1, w2 in newRelations]
				feed.extend(reverseRelation)
	if(len(feed) == 0):
		# Out of data, quitting
		return None
	batch = feed[:batchSize]
	inputFeed = []
	outputFeed = []
	# print(batch)
	for i, o in batch:
		# if cbowMode, input is already array, if skip-gram then it need to be converted into array
		inputFeed.append(i if cbowMode else [i])
		outputFeed.append([o])
	del feed[:batchSize]
	
	# if last batch, add dummy data to feed into inputs
	if(len(inputFeed) < batchSize):
		inputSize = len(inputFeed[0])
		for i in range(batchSize-len(inputFeed)):
			if(cbowMode):
				inputFeed.append([0] * inputSize)
			else:
				inputFeed.append([0])
			outputFeed.append([0])
	
	return inputFeed, outputFeed
	
def generateDictionaryFromParser(lines, regex, regexGroupIdx, useMatch=True):
	dict = {}
	for line in lines:
		match = re.match(regex, line) if(useMatch) else re.search(regex, line)
		if(match is not None):
			if(useMatch):
				word = match.group(regexGroupIdx)
				dict[word] = dict.get(word, 0) + 1
				# print(word)
			else:
				for arr in match:
					word = arr[regexGroupIdx]
					dict[word] = dict.get(word, 0) + 1
	
	return dict
	
def generateDictionaryFromLines(lines, lowercase=False, decapitalize=False):
	dict = {}
	for line in lines:
		if(lowercase):
			line = line.lower()
		words = line.strip().split()
		for word in words:
			if(decapitalize and checkCapitalize(word)):
				word = word.lower()
				dict[capitalize_token] = dict.get(capitalize_token, 0) + 1
			dict[word] = dict.get(word, 0) + 1
	return dict
	
def organizeDict(wordCountDict, tagDict, dictSize, extraWords=["*UNKNOWN*"]):
	# Dict will be sorted from highest to lowest appearance
	if(dictSize < 0):
		# Obscure mode, remove all items that show up less than |threshold|
		threshold = abs(dictSize)
		wordCountDict = {k: v for k, v in wordCountDict.items() if v > threshold}
		print("Found {:d} words that fit the criteria: appeared more than {:d} times".format(len(wordCountDict), threshold))
	listWords = [w for w,c in sorted(wordCountDict.items(), key=lambda item: item[1], reverse=True)]
	listWords = listWords[:dictSize-1] if(dictSize > 0) else listWords
	listWords = extraWords + listWords
	for key in tagDict:
		listWords.append(tagChangeFunc(key))
	# create two dictionary for reference
	wordDict = {}
	refDict = {}
	for i in range(len(listWords)):
		word = listWords[i]
		wordDict[word] = i
		refDict[i] = word
	
	return len(listWords), wordDict, refDict

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
		allWordDict = generateDictionaryFromLines(fileIn.readlines(), lowercase=lowercase, decapitalize=decapitalize)
	else:
		allWordDict = generateDictionaryFromParser(fileIn.readlines(), conllRegex, 2)
	print("All words found: {}".format(len(allWordDict)))
	dictSize, wordDict, refDict = organizeDict(allWordDict, tagDict, dictSize, extraWords=extraWords)
	
	# Create a session based on the actual dictSize (plus unknownWord and tags and maybe start/stop sentence token)
	# Session will only accept this batch size from then on
	if(parseMode == 'extended'):
		embeddingInputSize = 2
	elif(embMode == 'cbow'):
		# input the size of window
		embeddingInputSize = wordWindow * 2
	else:
		embeddingInputSize = 1
	# extended must use cbow scheme to support its closeleft-left-closeright-right tokens despite being skipgram
	if((parseMode == 'extended' or embMode == 'cbow') and properCBOW):
		sessionTuple = createEmbeddingSessionCBOW(dictSize, embeddingSize, embeddingInputSize, batchSize)
	else:
		sessionTuple = createEmbeddingSession(dictSize, embeddingSize, embeddingInputSize, batchSize)
	
	dictTuple = dictSize, wordDict, refDict, tagDict
	
	session = sessionTuple[0]
	session.run(tf.global_variables_initializer())
	# print(np.array2string(sessionTuple[4][0].eval(session=session)))
	
	return sessionTuple, dictTuple

def createEmbeddingObsolete(fileAndMode, tagDict, dictSize, embeddingSize, batchSize, extraWords):
	file, isNormal, isCBOW, windowSize, properCBOW, lowercase = fileAndMode
	# Destroy any current graph and session
	tf.reset_default_graph()
	session = tf.get_default_session()
	if(session is not None):
		session.close()
	
	file.seek(0)
	# Generate frequency dictionary
	if(isNormal):
		allWordDict = generateDictionaryFromLines(file.readlines(), lowercase)
	else:
		allWordDict = generateDictionaryFromParser(file.readlines(), conllRegex, 2)
	print("All words found: {}".format(len(allWordDict)))
	dictSize, wordDict, refDict = organizeDict(allWordDict, tagDict, dictSize, extraWords=extraWords)
	
	# Create a session based on the actual dictSize (plus unknownWord and tags and maybe start/stop sentence token)
	# Session will only accept this batch size from then on
	if(isCBOW):
		# input the size of window
		embeddingInputSize = wordWindow * 2
	else:
		embeddingInputSize = 1
	if(isNormal and isCBOW and properCBOW):
		sessionTuple = createEmbeddingSessionCBOW(dictSize, embeddingSize, embeddingInputSize, batchSize)
	else:
		sessionTuple = createEmbeddingSession(dictSize, embeddingSize, embeddingInputSize, batchSize)
	
	dictTuple = dictSize, wordDict, refDict, tagDict
	
	session = sessionTuple[0]
	session.run(tf.global_variables_initializer())
	# print(np.array2string(sessionTuple[4][0].eval(session=session)))
	
	return sessionTuple, dictTuple

def generateTrainingData(parseMode, embMode, fileIn, wordDict, batchSize, cbowGrandparent=False, lowercase=False, decapitalize=False):
	fileIn.seek(0)
	cbowMode = (embMode == 'cbow')
	if(parseMode == 'dependency'):
		dataBlock = getDataBlock(file.readlines(), blankLineRegex)
		dataBlock.pop(0)
		
	# generator function from this block to avoid memory overflow
		def getFeed():
			if(len(dataBlock) <= 0):
				return None
			block = dataBlock.pop()
			tree = constructTreeFromBlock(block, conllRegex)
			del block
			return createFeedData(tree, wordDict, (cbowMode, cbowGrandparent))
	elif(parseMode == 'normal' or parseMode == 'extended'):
		sentenceToDataFunc = createFeedDataFromSentence if parseMode == 'normal' else createExtendedFeedDataFromSentence
		sentences = parseSentencesFromLines(file.readlines(), (start_token, end_token, wordWindow) if cbowMode else False, lowercase, decapitalize=decapitalize)
		def getFeed():
			try:
				sentence = next(sentences)
			except StopIteration:
				return None
			return sentenceToDataFunc(sentence, wordDict, cbowMode, wordWindow)
#			return createFeedDataFromSentence(sentence, wordDict, cbowMode, wordWindow)
	else:
		raise Exception("Invalid parseMode: {:s}".format(parseMode))

	# if extended, force the cbowMode variable since it is formatted as cbow and we don't want the mirror to happen
	cbowMode = cbowMode or parseMode == 'extended'
	# initialize
	feed = getFeed()
	batch = getBatchFromFeedData(batchSize, feed, getFeed, cbowMode)
	# loop until no more feed/batch can be generated
	workingBatches = []
	while(batch is not None):
		workingBatches.append(batch)
		batch = getBatchFromFeedData(batchSize, feed, getFeed, cbowMode)
	
	return workingBatches

def trainEmbedding(createdData, sessionTuple, epoch, timerFunc=None, timerInterval=1000, passedSteps=0):
	session, train_op, training_inputs, training_outputs, resultTuple = sessionTuple
	for i in range(1, epoch+1):
		total_loss = 0.0
		for inputBatch, outputBatch in createdData:
			_, loss = session.run(fetches=train_op, feed_dict={training_inputs:inputBatch, training_outputs:outputBatch})
			passedSteps += 1
			total_loss += loss
			if(passedSteps % timerInterval == 0 and timerFunc is not None):
				print("Steps {:d}; time passed {:.2f}s, last loss {:.4f}.".format(passedSteps, timerFunc(None), loss))
		currentTime = timerFunc(i)
		print("Epoch {:d} completed, time passed {:.2f}s, total loss in total / per batch: {:.5f} / {:.5f}".format(i, currentTime, total_loss, total_loss / float(len(createdData))))
	return passedSteps

def trainEmbeddingObsolete(fileAndMode, sessionTuple, dictTuple, batchSize, epoch, savedTrainData=None, timerFunc=None):
	# If in first iteration, disregard the epoch counter
	embeddingMode = fileAndMode[0]
	session, train_op, training_inputs, training_outputs, resultTuple = sessionTuple
	dictSize, wordDict, refDict, tagDict = dictTuple
	if(embeddingMode == 'dependency'):
		embeddingMode, file, cbowMode, cbowGrandparent = fileAndMode
		if(savedTrainData == None or savedTrainData == True):
			# Do not save data / first time generate data
			# Conll format, generate dependency tree block and create batch data
			savedBatches = []
			#return to head of the file, this time split in blocks and create appropriate tree
			file.seek(0)
			
			dataBlock = getDataBlock(file.readlines(), blankLineRegex)
			dataBlock.pop(0)
			
			# generator function for the getBatch process
			def getFeedFromBlock():
				if(len(dataBlock) <= 0):
					return None
				block = dataBlock.pop()
				tree = constructTreeFromBlock(block, conllRegex)
				del block
				return createFeedData(tree, wordDict, (cbowMode, cbowGrandparent))
			
			# initialize
			feed = getFeedFromBlock()
			batch = getBatchFromFeedData(batchSize, feed, getFeedFromBlock, cbowMode)
			# loop until no more feed/batch can be generated
			while(batch is not None):
				if(savedTrainData):
					savedBatches.append(batch)
				# Handle data made by batch
				inputFeed, outputFeed = batch
				session.run(fetches=train_op, feed_dict={training_inputs: inputFeed, training_outputs: outputFeed})
				# Load new batch
				batch = getBatchFromFeedData(batchSize, feed, getFeedFromBlock, cbowMode)
				
			# Saved data will be returned
			return savedBatches
		elif(isinstance(savedTrainData, list)):
			# use saved data to train
			loss = 0.0
			for i in range(1, epoch+1):
				if(i % 100 == 0 and timerFunc is not None):
					print("Iteration %d, time passed %.2fs, total loss per iter %.2f" % (i, timerFunc(i), loss / 100))
					# reset loss count
					loss = 0.0
				for batch in savedTrainData:
					_, newloss = session.run(fetches=train_op, feed_dict={training_inputs: batch[0], training_outputs: batch[1]})
					loss += newloss
		else:
			raise Exception("Wrong input of savedTrainData (neither list nor boolean)")
	elif(embeddingMode == 'normal'):
		cbowMode = fileAndMode[2]
		if(cbowMode):
			embeddingMode, file, cbowMode, startToken, endToken, wordWindow, lowercase = fileAndMode
		else:
			embeddingMode, file, cbowMode, wordWindow, lowercase = fileAndMode
		if(savedTrainData == None or savedTrainData == True):
			savedBatches = []
			file.seek(0)
			sentences = parseSentencesFromLines(file.readlines(), (start_token, end_token, wordWindow) if cbowMode else False, lowercase=lowercase, decapitalize=decapitalize)
			
			# generator function for getBatch
			def getFeedFromSentence():
				try:
					sentence = next(sentences)
				except StopIteration:
					return None
				return createFeedDataFromSentence(sentence, wordDict, cbowMode, wordWindow)
			
			feed = getFeedFromSentence()
			batch = getBatchFromFeedData(batchSize, feed, getFeedFromSentence, cbowMode)
			while(batch is not None):
				if(savedTrainData):
					savedBatches.append(batch)
				# Handle data made by batch
				inputFeed, outputFeed = batch
				session.run(fetches=train_op, feed_dict={training_inputs: inputFeed, training_outputs: outputFeed})
				# Load new batch
				batch = getBatchFromFeedData(batchSize, feed, getFeedFromSentence, cbowMode)
				
			# Saved data will be returned
			return savedBatches
		elif(isinstance(savedTrainData, list)):
			# use saved data to train
			loss = 0.0
			for i in range(1, epoch+1):
				if(i % 100 == 0 and timerFunc is not None):
					print("Iteration %d, time passed %.2fs, total loss per iter %.2f" % (i, timerFunc(i), loss / 100))
					# reset loss count
					loss = 0.0
				for batch in savedTrainData:
					_, newloss = session.run(fetches=train_op, feed_dict={training_inputs: batch[0], training_outputs: batch[1]})
					loss += newloss
		else:
			raise Exception("Wrong input of savedTrainData (neither list nor boolean)")

def evaluateSimilarity(divResult):
	return 1 - np.tanh(np.abs(np.log(divResult)))

def findSimilarity(tupleMatrix, refDict, sample, mode=3):
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
	findSimilarity((embeddingMatrix, normalizedMatrix), refDict, random_sample)

	return embeddingMatrix, normalizedMatrix

def findAndPrintNearest(tupleMatrix, combinedDict, wordList, mode=3):
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
	
	findSimilarity(tupleMatrix, refDict, sample, mode=mode)

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
		file = io.open(outputDir + '.' + outputExt, 'w', encoding = 'utf-8')
		file.write(embeddingCountAndSize)
		for word in wordDict:
			if(tagDictForRemoval and word in tagDictForRemoval):
				continue
			idx = wordDict[word]
			writeWordToFile(file, word, resultMatrix[idx])
		file.close()
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
		file = io.open(outputDir + '.' + outputExt, "wb")
		pickle.dump(normalDict if exportMode == "binary" else (normalDict, normalizedDict), file)
		file.close()
	else:
		raise Exception("Wrong mode @exportEmbedding, must be all|both|default|normalized|binary|binary_full")

def writeWordToFile(file, word, vector, dimensionFormat="%.6f"):
	file.write(word + '\t')
	first = True
	for d in vector:
		if(not first):
			file.write(' ')
		else:
			first = False
		file.write(dimensionFormat % d)
	file.write('\n')

def writeListWordsToFile(fileOrFileDir, wordDict):
	if(isinstance(fileOrFileDir, str)):
		file = io.open(fileOrFileDir, 'w', encoding = 'utf-8')
	else:
		file = fileOrFileDir
	for word in wordDict:
		file.write(word + '\n')
	file.close()
	
def modeStringParse(string):
	# mode is basically (isNormal, isCBOW, windowsize) tuple here
	string = string.lower().split('_')
	if(any(string[0] == item for item in ['normal', 'dependency', 'extended'])):
		isNormal = (string[0] == 'normal')
	else:
		raise argparse.ArgumentTypeError('Arg1 must be dependency/normal/extended')
	
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
		if(not args.terminal_commands_only and items is not None):
			resultTuple, dictTuple = items
		elif(args.terminal_commands_only):
			print("Warning: in terminal command only mode, you do not have data. You must load a previous dump.")
		else:
			raise ValueError("Items is None while not args.terminal_commands_only.")
		print("+++ Console Interactions Ready +++")
		helperLine = "q/quit to exit, m/mode to select comparing embedding, e/export to try export a distribution image, i/input to input a list of words for comparison, s/save and l/load to save or load the command to directory"
		print(helperLine)
		selectMode = 3
		while(True):
			lastCommand = input("Input command: ").strip()
			if(lastCommand == 'quit' or lastCommand == 'q'):
				break
			elif(lastCommand == 'mode' or lastCommand == 'm'):
				try:
					selectMode = int(input("1 for full, 2 for normalized, 3 for both: "))
					if(selectMode > 3 or selectMode < 0):
						selectMode = 3
						print("Must select from the options specified.")
					else:
						print("Mode(int) changed to {:d}".format(selectMode))
				except ValueError:
					print("Invalid value inputted, must be int")
			elif(lastCommand == 'export' or lastCommand == 'e'):
				print("Unimplemented, please choose another one")
			elif(lastCommand == 'help' or lastCommand == 'h'):
				print(helperLine)
			elif(lastCommand == 'input' or lastCommand == 'i'):
				wordOrWords = input("The comparing words: ").strip()
				if(wordOrWords.find(" ") >= 0):
					listWords = wordOrWords.split()
				else:
					listWords = [wordOrWords]
				findAndPrintNearest(resultTuple, dictTuple, listWords, mode=selectMode)
			elif(lastCommand == 'save' or lastCommand == 's'):
				path = input("Specify the save path: ")
				saveFile = io.open(path, "wb")
				pickle.dump((resultTuple, dictTuple), saveFile)
				print("Dumped the needed data @{:s}".format(path))
			elif(lastCommand == 'load' or lastCommand == 'l'):
				path = input("Specify the load path: ")
				loadFile = io.open(path, "rb")
				resultTuple, dictTuple = pickle.load(loadFile)
				print("Loaded the needed data @{:s}".format(path))
			else:
				print("Invalid command, type help/h for a list of valid commands.")

if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Create training examples from resource data.')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input files')
	parser.add_argument('-m', '--mode', type=modeStringParse, required=True, help='the mode to embed the word2vec in, must be in format (dependency|normal|extended)_(skipgram|cbow)_wordWindow(only if in normal mode)')
	parser.add_argument('-x', '--export_mode', required=True, type=str, help='exporting the values to an outside file, must be (all|both|default|normalized|binary|binary_full|vocab)')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-t','--tagdir', type=str, default="all_tag.txt", help='location of the tag file containing both POStag and dependency, default all_tag.txt')
	parser.add_argument('--input_extension', type=str, default="conllx", help='file extension for input file, default conllx')
	parser.add_argument('--output_extension', type=str, default=None, help='file extension for output file, default embedding.txt/embedding.bin')
	parser.add_argument('--unknown_word', type=str, default="*UNKNOWN*", help='placeholder name for words not in dictionary, default *UNKNOWN*')
	parser.add_argument('--epoch', type=int, default=1000, help='number of iterations through the data, default 1000')
	parser.add_argument('--batch_size', type=int, default=512, help='size of the batches to be feed into the network, default 512')
	parser.add_argument('--embedding_size', type=int, default=100, help='size of the embedding to be created, default 100')
	parser.add_argument('--dict_size', type=int, default=10000, help='size of the words to be embedded, default 10000, input -1 for all words, -n (int) for taking only those occurred more than n times.')
	parser.add_argument('-e','--evaluate', type=int, default=0, help='try to evaluate the validity of the trained embedding. Note that at the end of the training the evaluation function will fire regardless of this value. A positive number for the steps where you launch the evaluation')
	parser.add_argument('--filter_tag', action='store_true', help='remove the trained tag from the output file')
	parser.add_argument('--grandparent', action='store_true', help='use grandparent scheme, only available to dependency_cbow mode')
	parser.add_argument('--average', action='store_false', help='use average tensor instead of fully independent tensor, only available to normal_cbow mode')
	parser.add_argument('--timer', type=int, default=1000, help='the inteval to call timer func')
	parser.add_argument('--lowercase', action='store_true', help='do the lowercase by the default python function. Not recommended.')
	parser.add_argument('--decapitalize', action='store_true', help='create a <cap> token before capitalized words')
	parser.add_argument('--terminal_commands', action='store_true', help='run a terminal after training to check on the result')
	parser.add_argument('--terminal_commands_only', action='store_true', help='run a terminal only, doing nothing about the script itself')
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
	isNormal, isCBOW, wordWindow, modeString = args.mode
	parseMode, embMode, _ = modeString
	
	timer = time.time()
	
	file = io.open(args.inputdir + '.' + args.input_extension, 'r', encoding='utf-8')
	tagDict = getTagFromFile(args.tagdir, True)
	# Exit prematurely with vocab export
	if(args.export_mode == 'vocab'):
		allWordDict = generateDictionaryFromLines(file.readlines(), lowercase=args.lowercase, capitalize=args.decapitalize)
		writeListWordsToFile(args.outputdir + '.' + args.output_extension, allWordDict)
		print("Done for vocab export, time passed %.2fs" % (time.time() - timer))
		sys.exit(0)
	# Initialize the embedding
	# Todo add the inverse mode
	if(parseMode == 'normal'):
		# add the sos and eos token as well
		extraWords = [unknownWord, start_token, end_token]
	elif(parseMode == 'extended'):
		# add far left - close left - close right - far right
		extraWords = [unknownWord, far_left_token, close_left_token, close_right_token, far_right_token]
	else: #elif(parseMode == 'dependency'):
		# add only the unknownWord
		extraWords = [unknownWord]
	sessionTuple, dictTuple = createEmbedding(parseMode, embMode, file, (dictSize, wordWindow, embeddingSize, batchSize), tagDict=tagDict, properCBOW=args.average, extraWords=extraWords, lowercase=args.lowercase, decapitalize=args.decapitalize)
#	sessionTuple, dictTuple = createEmbedding((file, isNormal, isCBOW, wordWindow, args.average, args.lowercase), tagDict, dictSize, embeddingSize, batchSize, unknownWord if(not isNormal) else [unknownWord, '<s>', '<\s>'])
	print("Done for @createEmbedding, time passed %.2fs" % (time.time() - timer))
	
	# create the static sample for evaluation
	static_sample = np.random.choice(sampleWordWindow, sampleSize, replace=False)
	# Train and evaluate the embedding
	def timerFunc(counter=None):
		if(args.evaluate > 0 and counter is not None):
			if(counter % args.evaluate == 0):
				evaluateEmbedding(sessionTuple, dictTuple, sampleSize, sampleWordWindow, checkSize, sample=static_sample)
		return time.time() - timer
		
	wordDict = dictTuple[1]
	generatedTrainData = generateTrainingData(parseMode, embMode, file, wordDict, batchSize, cbowGrandparent=args.grandparent, lowercase=args.lowercase)
	print("Done generating training data, time passed {:.2f}s, generated batch of size {:d}".format(time.time() - timer, len(generatedTrainData)))
	
	totalSteps = trainEmbedding(generatedTrainData, sessionTuple, epoch, timerFunc=timerFunc, timerInterval=args.timer)
	print("All training complete @trainEmbedding, total steps {:d}, time passed {:.2f}s".format(totalSteps, time.time() - timer))
#	savedTrainData = trainEmbedding(fileAndMode, sessionTuple, dictTuple, batchSize, 1, True)
#	print("Done generating @trainEmbedding (first iteration), data included %d batches(size %d), time passed %.2fs" % (len(savedTrainData), batchSize, time.time() - timer))
#	trainEmbedding(fileAndMode, sessionTuple, dictTuple, batchSize, epoch, savedTrainData, timerFunc)
#	print("Done full training by @trainEmbedding (%d iteration), time passed %.2fs" % (epoch, time.time() - timer))
	
	# Final evaluation. Must run, to bring out the resultTuple
	resultTuple = evaluateEmbedding(sessionTuple, dictTuple, sampleSize, sampleWordWindow, checkSize)
	print("Final @evaluateEmbedding on random sample, time passed %.2fs" % (time.time() - timer))
	
	dictSize = dictTuple[0]
	embeddingCountAndSize = "{} {}\n".format(dictSize, embeddingSize)
	
	# Export to file
	exportEmbedding(args.export_mode, args.outputdir, args.output_extension, dictTuple[1], resultTuple, embeddingCountAndSize)
	
	file.close()

	if(args.terminal_commands):
		# Allow reading words from the terminal and output closest words found
		runTerminal(args, (resultTuple, dictTuple))
