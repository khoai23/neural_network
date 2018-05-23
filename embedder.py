import tensorflow as tf
import sys, re, argparse, io, time, pickle
from ffBuilder import *
from exampleBuilder import *
import numpy as np

conllRegex = re.compile("(\d+)\t(.+)\t_\t([\w$\.,:\"\'-]+)\t([\w$\.,:\"\'-]+)\t_\t(\d+)\t(\w+)")
WORD_WINDOW = 2
def tagChangeFunc(tag):
	return tag + '_<CH>'

def parseSentencesFromLines(lines, appendToken=False, lowercase=False):
	if(not lowercase):
		sentences = (line.strip().split(' ') for line in lines if(line.strip() != ""))
	else:
		sentences = (line.strip().lower().split(' ') for line in lines if(line.strip() != ""))
	if(appendToken):
		startToken, endToken, window = appendToken
		startToken = [startToken] * window
		endToken = [endToken] * window
		def modifySentence(sentence):
			return startToken + sentence + endToken
		return (modifySentence(sentence) for sentence in sentences)
	else:
		return sentences
	
def getRelationFromParentNode(parentNode, listRelation, lookupDict):
	for child in parentNode.children:
		# relation will include dependency to parent, grandparent, parent, child, dependency to child, relating in that order
		# will create relation in tuple of two, and require reversing after
		relations = []
		grandparent = parentNode.parent
		if(grandparent is not None):
			relations.append((lookupDict.get(tagChangeFunc(parentNode.dependency), 0), lookupDict.get(grandparent.word, 0)))
			relations.append((lookupDict.get(grandparent.word,0), lookupDict.get(parentNode.word, 0)))
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
			listRelation.append( ([lookupDict.get(parentNode.word, defaultWordIdx), lookupDict.get(tagChangeFunc(parentNode.dependency, defaultWordIdx)),
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
	
def generateDictionaryFromLines(lines, lowercase=False):
	dict = {}
	for line in lines:
		if(lowercase):
			line = line.lower()
		words = line.strip().split()
		for word in words:
			dict[word] = dict.get(word, 0) + 1
	return dict
	
def organizeDict(wordCountDict, tagDict, dictSize, unknownWord="*UNKNOWN*"):
	# Dict will be sorted from highest to lowest appearance
	listWords = [w for w,c in sorted(wordCountDict.items(), key=lambda item: item[1], reverse=True)]
	listWords = listWords[:dictSize-1] if(dictSize > 0) else listWords
	if(isinstance(unknownWord, str)):
		listWords.insert(0, unknownWord)
	else:
		listWords = unknownWord + listWords
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

def createEmbedding(fileAndMode, tagDict, dictSize, embeddingSize, batchSize, extraWords):
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
		print("All words found: {}".format(len(allWordDict)))
		dictSize, wordDict, refDict = organizeDict(allWordDict, tagDict, dictSize, extraWords)
	else:
		allWordDict = generateDictionaryFromParser(file.readlines(), conllRegex, 2)
		print("All words found: {}".format(len(allWordDict)))
		dictSize, wordDict, refDict = organizeDict(allWordDict, tagDict, dictSize, extraWords)
	
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

def trainEmbedding(fileAndMode, sessionTuple, dictTuple, batchSize, epoch, savedTrainData=None, timerFunc=None):
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
			global blockIdx
			blockIdx = 0
			dataBlock = getDataBlock(file.readlines(), blankLineRegex)
			
			# generator function for the getBatch process
			def getFeedFromBlock():
				global blockIdx
				if(blockIdx >= len(dataBlock)):
					return None
				block = dataBlock[blockIdx]
				tree = constructTreeFromBlock(block, conllRegex)
				blockIdx += 1
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
			sentences = parseSentencesFromLines(file.readlines(), (startToken, endToken, wordWindow) if cbowMode else False, lowercase)
			
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

def evaluateEmbedding(sessionTuple, refDict, sampleSize, sampleWordWindow, checkSize):
	# Taken straight from the basic word2vec
	# use numpy instead of tf.matmul
	session=sessionTuple[0]
	
	# sampleSize samples in range of 0-wordWindow, no duplication
	random_sample = np.random.choice(sampleWordWindow, sampleSize, replace=False)
	
	# get the un-normalized version and check for closest values
	_, _, _, _, resultTuple = sessionTuple
	embeddingMatrix = resultTuple[0].eval(session=session)
	normalizedMatrix = resultTuple[1].eval(session=session)
	# print(np.array2string(normalizedMatrix[2:6]))
	sampleMatrix = np.transpose([normalizedMatrix[i] for i in random_sample])
	# print([[i for i in vector] for vector in sampleMatrix])
	
	# print(embeddingMatrix.shape, vectorLength.shape)
	# calculate distance for normalized version
	distance = np.transpose(np.matmul(normalizedMatrix, sampleMatrix))
	print("Check normalized version:")
	for idx in range(sampleSize):
		word = refDict[random_sample[idx]]
		nearest = (-distance[idx, :])
		# print(nearest)
		nearest = nearest.argsort()[:checkSize + 1]
		nearest = [refDict[x] for x in nearest]
		print("{} -> {}".format(word, nearest))
	
	vectorLength = np.linalg.norm(embeddingMatrix, axis=1)
	magDifference = np.asarray([vectorLength / vectorLength[sample] for sample in random_sample])
	magDifference = 1 - np.tanh(np.abs(np.log(magDifference)))
	# multiply the normalized part with the difference of magnitude converted into 0-1 range (1 is the same, 0 is infinitively different)
	print("Check normal version:")
	#print(distance.shape, magDifference.shape)
	distance = np.multiply(magDifference, distance)
	for idx in range(sampleSize):
		word = refDict[random_sample[idx]]
		nearest = -distance[idx]
		nearest = nearest.argsort()[:checkSize + 1]
		nearest = [refDict[x] for x in nearest]
		print("{} -> {}".format(word, nearest))
	
	return embeddingMatrix, normalizedMatrix

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
		use_dict = 0 if(exportMode == "default") else 1
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
		# Only export the normal version in normal binary, since the normalized one can be made from the full
		# use pickle
		normalDict = {}
		if("full" in exportMode):
			normalizedDict = {}
		assert len(wordDict) == len(resultMatrixTuple[0]) and len(wordDict) == len(resultMatrixTuple[1])
		for word in wordDict:
			idx = wordDict[word]
			normalDict[word] = resultMatrixTuple[0][idx]
			if("full" in exportMode):
				normalizedDict[word] = resultMatrixTuple[1][idx]
		file = open(outputDir + '.' + outputExt, "wb" )
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

def modeStringToTuple(str):
	# mode is basically (isNormal, isCBOW) tuple
	str = str.split('_')
	if(str[0] == 'dependency'):
		isNormal = False
	elif(str[0] == 'normal'):
		isNormal = True
	else:
		raise argparse.ArgumentTypeError('Arg2 must be skipgram/cbow')
	
	if(str[1] == 'skipgram'):
		isCBOW = False
	elif(str[1] == 'cbow'):
		isCBOW = True
	else:
		raise argparse.ArgumentTypeError('Arg2 must be skipgram/cbow')
	
	try:
		windowSize = int(str[3])
	except Exception as e:
		print("Error getting windowSize (%s), default to %d" % (e, WORD_WINDOW))
		windowSize = WORD_WINDOW
	return isNormal, isCBOW, windowSize
	
if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Create training examples from resource data.')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input files')
	parser.add_argument('-m', '--mode', type=modeStringToTuple, required=True, help='the mode to embed the word2vec in, must be in format (dependency|normal)_(skipgram|cbow)_wordWindow(only if in normal mode)')
	parser.add_argument('-x', '--export_mode', required=True, type=str, help='exporting the values to an outside file, must be (all|both|default|normalized|binary|binary_full)')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-t','--tagdir', type=str, default="all_tag.txt", help='location of the tag file containing both POStag and dependency, default all_tag.txt')
	parser.add_argument('--input_extension', type=str, default="conllx", help='file extension for input file, default conllx')
	parser.add_argument('--output_extension', type=str, default=None, help='file extension for output file, default embedding.txt/embedding.bin')
	parser.add_argument('--unknown_word', type=str, default="*UNKNOWN*", help='placeholder name for words not in dictionary, default *UNKNOWN*')
	parser.add_argument('--epoch', type=int, default=1000, help='number of iterations through the data, default 1000')
	parser.add_argument('--batch_size', type=int, default=512, help='size of the batches to be feed into the network, default 512')
	parser.add_argument('--embedding_size', type=int, default=100, help='size of the embedding to be created, default 100')
	parser.add_argument('--dict_size', type=int, default=10000, help='size of the words to be embedded, default 10000, input -1 for all words')
	parser.add_argument('-e','--evaluate', type=int, default=0, help='try to evaluate the validity of the trained embedding. -1 for not evaluating, 0 for evaluating at the end of the training, and positive number for the steps where you launch the evaluation')
	parser.add_argument('--filter_tag', action='store_true', help='remove the trained tag from the output file')
	parser.add_argument('--grandparent', action='store_true', help='use grandparent scheme, only available to dependency_cbow mode')
	parser.add_argument('--average', action='store_false', help='use average tensor instead of fully independent tensor, only available to normal_cbow mode')
	parser.add_argument('--timer', type=int, default=100, help='the inteval to call timer func')
	parser.add_argument('--lowercase', action='store_true', help='do the lowercase by the default python function. Not recommended.')
	parser.add_argument('--other_mode', action='store_true', help='placeholder')
	args = parser.parse_args()
	
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
	isNormal, isCBOW, wordWindow = args.mode
	
	timer = time.time()
	
	file = io.open(args.inputdir + '.' + args.input_extension, 'r', encoding='utf-8')
	tagDict = getTagFromFile(args.tagdir, True)
	
	# Initialize the embedding
	sessionTuple, dictTuple = createEmbedding((file, isNormal, isCBOW, wordWindow, args.average, args.lowercase), tagDict, dictSize, embeddingSize, batchSize, \
												unknownWord if(not isNormal) else [unknownWord, '<s>', '<\s>'])
	print("Done for @createEmbedding, time passed %.2fs" % (time.time() - timer))
	
	# Train and evaluate the embedding
	if(isNormal):
		if(isCBOW):
			fileAndMode = 'normal', file, isCBOW, '<s>', '<\s>', wordWindow, args.lowercase
		else:
			fileAndMode = 'normal', file, isCBOW, wordWindow, args.lowercase
	else:
		fileAndMode = 'dependency', file, isCBOW, args.grandparent
	def timerFunc(counter=None):
		if(args.evaluate > 0 and counter is not None):
			if(counter % args.evaluate == 0):
				evaluateEmbedding(sessionTuple, dictTuple[2], sampleSize, sampleWordWindow, checkSize)
		return time.time() - timer
		
	savedTrainData = trainEmbedding(fileAndMode, sessionTuple, dictTuple, batchSize, 1, True)
	print("Done generating @trainEmbedding (first iteration), data included %d batches(size %d), time passed %.2fs" % (len(savedTrainData), batchSize, time.time() - timer))
	trainEmbedding(fileAndMode, sessionTuple, dictTuple, batchSize, epoch, savedTrainData, timerFunc)
	print("Done full training by @trainEmbedding (%d iteration), time passed %.2fs" % (epoch, time.time() - timer))
	
	# Final evaluation
	if(args.evaluate >= 0):
		resultTuple = evaluateEmbedding(sessionTuple, dictTuple[2], sampleSize, sampleWordWindow, checkSize)
		print("Final @evaluateEmbedding, time passed %.2fs" % (time.time() - timer))
	
	dictSize = dictTuple[0]
	embeddingCountAndSize = "{} {}\n".format(dictSize, embeddingSize)
	
	# Export to file
	exportEmbedding(args.export_mode, args.outputdir, args.output_extension, dictTuple[1], resultTuple, embeddingCountAndSize)
	
	file.close()