import ffBuilder as builder
import numpy as np
import tensorflow as tf
import sys, os, pickle, argparse, io, time, random
from calculatebleu import *

def getEmbeddingFromFile(fileDir, useDefault=True):
	file = io.open(fileDir, 'rb')
	dictTuple = pickle.load(file)
	file.close()
	if(isinstance(dictTuple, dict)):
		return dictTuple
	elif(isinstance(dictTuple, (tuple, list))):
		return dictTuple[0 if useDefault else 1]
	else:
		raise Exception("Wrong type during pickle read file")
		
def getSentencesFromFile(fileDir, splitToken=' '):
	file = io.open(fileDir, 'r', encoding='utf-8')
	lines = file.readlines()
	file.close()
	result = []
	for line in lines:
		if(line.find(splitToken) >= 0):
			result.append(line.split(splitToken))
		else:
			result.append([line])
	return result
	
def createSentenceCouplingFromFile(args):
	srcSentences = getSentencesFromFile(os.path.join(args.directory, args.src_file))
	tgtSentences = getSentencesFromFile(os.path.join(args.directory, args.tgt_file))
	assert len(srcSentences) == len(tgtSentences)
	print('Sentences read from file: %d' % len(srcSentences))
	coupling = []
	for i in range(len(srcSentences)):
		# filter out those which are too long
		if(len(srcSentences[i]) <= args.maximum_sentence_length and len(tgtSentences[i]) <= args.maximum_sentence_length):
			coupling.append((srcSentences[i], tgtSentences[i]))
		
	return coupling

def createEmbeddingCouplingFromFile(args):
	srcDict = getEmbeddingFromFile(os.path.join(args.directory, args.src_dict_file), args.import_default_dict)
	tgtDict = getEmbeddingFromFile(os.path.join(args.directory, args.tgt_dict_file), args.import_default_dict)
	# Convert the src/tgt dict into normal (word to id), ref (id to word), embeddingVector (id to vector)
	counter = 0
	srcWordToId, srcIdToWord = {}, {}
	srcEmbeddingVector = []
	for key in srcDict:
		assert len(srcEmbeddingVector) == counter
		srcWordToId[key] = counter
		srcIdToWord[counter] = key
		srcEmbeddingVector.append(srcDict[key])
		counter += 1
	
	counter = 0
	tgtWordToId, tgtIdToWord = {}, {}
	tgtEmbeddingVector = []
	for key in tgtDict:
		assert len(tgtEmbeddingVector) == counter
		tgtWordToId[key] = counter
		tgtIdToWord[counter] = key
		tgtEmbeddingVector.append(tgtDict[key])
		counter += 1
		
	
	return (srcWordToId, srcIdToWord, np.array(srcEmbeddingVector)), (tgtWordToId, tgtIdToWord, np.array(tgtEmbeddingVector))

def createSession(args, embedding):
	isTrainingMode = args.mode == "train"
	srcEmbedding, tgtEmbedding = embedding
	srcEmbeddingDict, _, srcEmbeddingVector = srcEmbedding
	tgtEmbeddingDict, _, tgtEmbeddingVector = tgtEmbedding
	srcEmbeddingSize = srcEmbeddingVector.shape[1]
	tgtEmbeddingSize = tgtEmbeddingVector.shape[1]
	
	session = tf.Session()
	# dropout value, used for training. Must reset to 1.0(all) when using the decoding
	dropout = tf.placeholder_with_default(1.0, shape=())
	# input in shape (batchSize, inputSize) - not using timemayor
	input = tf.placeholder(shape=[None, None], dtype=tf.int32)
	# input are lookup from the known srcEmbeddingVector, shape (batchSize, inputSize, embeddingSize)
	srcEmbeddingVector = tf.constant(srcEmbeddingVector)#, dtype=tf.float32)
	inputVector = tf.nn.embedding_lookup(srcEmbeddingVector, input)
	# craft the encoder depend on the input vector. Currently using default values for all version
	settingDict = {'inputType':inputVector, 'layerSize':tgtEmbeddingSize, 'inputSize':None, 'dropout':dropout}
	inputFromEncoder, encoderOutput, encoderState, dropoutFromEncoder = builder.createEncoder(settingDict)
	assert inputFromEncoder is inputVector and dropoutFromEncoder is dropout
	# craft the output in shape (batchSize, outputSize)
	output = tf.placeholder(shape=[None, None], dtype=tf.int32)
	decoderInput = tf.placeholder(shape=[None, None], dtype=tf.int32)
	# the outputLengthList is the length of the sentence supposed to be output. Used to create loss function
	outputLengthList = tf.placeholder(shape=[None], dtype=tf.int32)
	# These are the dimension of the batch
	batchSize = tf.placeholder(shape=(), dtype=tf.int32)
	maximumUnrolling = tf.placeholder_with_default(args.maximum_sentence_length, shape=())
	# likewise, the output will be looked up into shape (batchSize, inputSize, embeddingSize)
	tgtEmbeddingVector = tf.constant(tgtEmbeddingVector)#, dtype=tf.float32)
	outputVector = tf.nn.embedding_lookup(tgtEmbeddingVector, output)
	decoderInputVector = tf.nn.embedding_lookup(tgtEmbeddingVector, decoderInput)
	# decoder will use the encoderState to work, outputVector and tgtEmbeddingVector for lookup check
	# also need a mode placeholder for switching between decoder helper and the start/end tokenId to search for 
	startTokenId, endTokenId = tgtEmbeddingDict['<s>'], tgtEmbeddingDict[args.end_token]
	# mode = tf.placeholder_with_default(True, shape=())
	# construct the settingDict
	settingDict['startTokenId'] = startTokenId; settingDict['endTokenId'] = endTokenId
	settingDict['correctResult'] = outputVector; settingDict['outputEmbedding'] = tgtEmbeddingVector
	settingDict['correctResultLen'] = outputLengthList; settingDict['encoderState'] = encoderState; settingDict['encoderOutputSize'] = tgtEmbeddingSize
	settingDict['batchSize'] = batchSize; settingDict['maximumDecoderLength'] = maximumUnrolling; settingDict['decoderInput'] = decoderInputVector
	logits, loss, decoderState, crossent = builder.createDecoder(isTrainingMode, settingDict)
	# TrainingOp function, built on the loss function
	if(isTrainingMode):
	#trainingTrainOp = builder.createOptimizer({'loss':loss[0], 'mode':'adam'})
		trainOp = builder.createOptimizer({'loss':loss, 'mode':'sgd', 'trainingRate':1.0})
	else:
		trainOp = None
	# initiate the session
	session.run(tf.global_variables_initializer())
	
	return session, [input, output, decoderInput], [batchSize, outputLengthList, maximumUnrolling, logits], [loss, trainOp]
	
def trainSessionOneBatch(args, sessionTuple, batch):
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	# unpack tensor placeholders to add to feed_dict
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, _ = configTuple
	# batch is formatted sets which had been padded into 2d (batchSize, maximumUnrolling) for both input/output
	# should be formattted as follow
	feed_dict = {input:batch[0], output:batch[1], batchSize:batch[2], outputLengthList:batch[3], maximumUnrolling:max(batch[3]), decoderInput:batch[4]}
	loss, _ = session.run(trainTuple, feed_dict=feed_dict)
	return loss
	
def trainSession(args, sessionTuple, batches, evaluationFunction=None):
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, _ = configTuple
	for step in range(args.epoch):
		args.global_steps += 1
		avgLosses = [0]
		for batch in batches:
			feed_dict = {input:batch[0], output:batch[1], batchSize:batch[2], outputLengthList:batch[3], maximumUnrolling:max(batch[3]), decoderInput:batch[4]}
			loss, _ = session.run(trainTuple, feed_dict=feed_dict)
			avgLosses[-1] += loss
		avgLosses[-1] = avgLosses[-1] / len(batches)
		if(evaluationFunction and (step+1) % args.evaluation_step == 0):
			# run evaluationFunction every evaluation_step epoch
			evaluationFunction((step+1,avgLosses))
		avgLosses.append(0)
	return avgLosses
	
def evaluateSession(args, session, dictTuple, sampleBatch):
	session, inputOutputTuple, configTuple, _ = sessionTuple
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, logits = configTuple
	_, _, tgtEmbeddingVector = dictTuple[1]
	feed_dict = {input:sampleBatch[0], batchSize:sampleBatch[2], outputLengthList:sampleBatch[3], maximumUnrolling:max(sampleBatch[3]), decoderInput:sampleBatch[4]}
	sampleResult = session.run(logits, feed_dict=feed_dict)
	_, savedData = getWordIdFromVectors(sampleResult[0], tgtEmbeddingVector, True)
	#print(resultInfer, resultTrain)
	# resultInfer = [getWordIdFromVectors(result, tgtEmbeddingVector, True, savedData) for result in resultInfer]
	sampleResult = [getWordIdFromVectors(result, tgtEmbeddingVector, True, savedData) for result in sampleResult]
	return sampleResult

def generateBatchesFromSentences(args, data, dictTuple):
	srcDictTuple, tgtDictTuple = dictTuple
	srcDict, tgtDict = srcDictTuple[0], tgtDictTuple[0]
	unknownWord = args.end_token
	srcUnknownID, tgtUnknownID = srcDict[unknownWord], tgtDict[unknownWord]
	# data are binding tuples of (s1, s2) for src-tgt, s1/s2 preprocessed into array of respective words
	batches = []
	srcBatch, tgtBatch = [], []
	for srcSentence, tgtSentence in data:
		srcSentence = [srcDict.get(word, srcUnknownID) for word in srcSentence]
		tgtSentence = [tgtDict.get(word, tgtUnknownID) for word in tgtSentence]
		srcBatch.append(srcSentence)
		tgtBatch.append(tgtSentence)
		if(len(srcBatch) == args.batch_size):
			# Full batch, begin converting
			assert len(srcBatch) == len(tgtBatch)
			padMatrix(srcBatch, srcDict[args.end_token])
			batchLengthList = padMatrix(tgtBatch, tgtDict[args.end_token])
			tgtInputBatch = [ ([tgtDict[args.start_token]] + list(tgt))[:-1] for tgt in tgtBatch]
			batchSize = args.batch_size
			batches.append((srcBatch, tgtBatch, batchSize, batchLengthList, tgtInputBatch))
			srcBatch, tgtBatch = [], []
	# Last batch
	padMatrix(srcBatch, srcDict[args.end_token])
	batchLengthList = padMatrix(tgtBatch, tgtDict[args.end_token])
	tgtInputBatch = [ ([tgtDict[args.start_token]] + list(tgt))[:-1] for tgt in tgtBatch]
	batchSize = len(srcBatch)
	batches.append((srcBatch, tgtBatch, batchSize, batchLengthList, tgtInputBatch))
	# Return the processed value
	return batches
	
def generateRandomBatchesFromSet(args, batches, paddingToken):
	# if batch too small, use first available
	inputPadding, outputPadding, outputStartToken = paddingToken
	if(len(batches) == 1):
		return batches[0]
	# batch_size list of where to take our sample
	# listSample = [random.randint(0, len(batches)-1) for _ in range(args.batch_size)]
	listSample = np.arange(len(batches))
	# get a random sample within that listSample
	listSample = [(i, random.randint(0, batches[i][2]-1)) for i in listSample]
	listSample = [(batches[i][0][j], batches[i][1][j], batches[i][3][j]) for i, j in listSample]
	# convert the sample back down to max size in listSample
	_, _, outputMaxLen = max(listSample, key=lambda s: s[2])
	bestLenInput, _, _ = max(listSample, key=lambda s:len(s[0]))
	inputMaxLen = len(bestLenInput)
	sampleInput, sampleOutput, sampleLengthList = [], [], []
	for input, output, length in listSample:
		if(len(output) > outputMaxLen):
			output = output[:outputMaxLen]
		elif(len(output) < outputMaxLen):
			output = output + [outputPadding] * (outputMaxLen-len(output))
		# print(len(input))
		if(len(input) < inputMaxLen):
			input = input + [inputPadding] * (inputMaxLen-len(input))
		sampleInput.append(input)
		sampleOutput.append(output)
		sampleLengthList.append(length)
	# construct the decoderInputBatch from tgtInputBatch
	# print([len(piece) for piece in sampleInput], inputMaxLen, np.array(sampleOutput).shape)
	sampleDecoderInput = [ ([outputStartToken] + list(out))[:-1] for out in sampleOutput]
	return sampleInput, sampleOutput, len(listSample), sampleLengthList, sampleDecoderInput
	
def padMatrix(matrix, paddingToken):
	# find the longest line in the matrix
	originalLength = [len(sentence) for sentence in matrix]
	maxLen = max(originalLength)
	# pad everything
	for sentence in matrix:
		if(len(sentence) < maxLen):
			sentence.extend([paddingToken] * (maxLen - len(sentence)))
	return originalLength

def getWordIdFromVectors(vectors, embedding, embeddingIsNormal=False, savedData=None):
	# only handle sentence-level vector
	# if normal, prepare vector length vs embeddingLength
	if(embeddingIsNormal):
		# getDistance data
		vectorLen = np.linalg.norm(vectors, axis=1)
		embeddingVectorLen = np.linalg.norm(embedding, axis=1) if savedData is None else savedData[1]
		lenDifference = [embeddingVectorLen / vectorLen[i] for i in range(len(vectorLen))]
		lenDifference = 1 - np.tanh(np.abs(np.log(lenDifference)))
		# Convert into normalized version
		vectors = np.divide(vectors.transpose(), vectorLen)
		# should be transposing embeddingVectorLen here, but it is 1-D so have to improvise
		embedding = np.transpose(np.divide(np.transpose(embedding), embeddingVectorLen)) if savedData is None else savedData[0]
		if(savedData is None):
			currentSaveData = embedding, embeddingVectorLen
	else:
		lenDifference = None
		# transpose to match vector to embedding, for matmul
		vectors = np.transpose(vectors)
	# matmul result in value in [-1;1] range represent the similarity between vectors
	vectorSimilarity = np.transpose(np.matmul(embedding, vectors))
	if(lenDifference is not None):
		assert vectorSimilarity.shape == lenDifference.shape
		vectorSimilarity = np.multiply(vectorSimilarity, lenDifference)
	# the highest in similarity is the id
	result = np.argmax(vectorSimilarity, axis=1)
	if('currentSaveData' in locals() and savedData is None):
		return result, currentSaveData
	return result
	
def createSessionStorageOps(session, opsAndTensors):
	listOfOpsRecorded, listOfTensorRecorded = opsAndTensors
	opsNames = []
	for op in listOfOpsRecorded:
		opsNames.append(op.name)
	tensorsNames = []
	for tensor in listOfTensorRecorded:
		tensorsNames.append(tensor.name)
	# print(opsNames, tensorsNames)
	
	tf.constant(opsNames, name="storageOps")
	tf.constant(tensorsNames, name="storageTensors")
def getOpsFromSessionStorage(session):
	storageOps = tf.get_default_graph().get_tensor_by_name("storageOps:0")
	storageTensors = tf.get_default_graph().get_tensor_by_name("storageTensors:0")
	opsNames, tensorsNames = session.run([storageOps, storageTensors])
	# apparently these string are saved as byte. WTF
	opsList = [tf.get_default_graph().get_operation_by_name(opName.decode()) for opName in opsNames]
	tensorsList = [tf.get_default_graph().get_tensor_by_name(tensorName.decode()) for tensorName in tensorsNames]
	return opsList, tensorsList

def sessionTupleToList(sessionTuple):
	# Organize tensor/ops into list
	_, inputOutputTuple, configTuple, trainTuple = sessionTuple
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, logits = configTuple
	loss, trainingOp = trainTuple
	return [trainingOp], [input, output, batchSize, outputLengthList, maximumUnrolling, logits[0], logits[1], loss]
def listToSessionTuple(opsAndTensor, session=None):
	# Convert from list back to tensor/ops
	opsList, tensorList = opsAndTensor
	trainingOp = opsList[0]
	input, output, batchSize, outputLengthList, maximumUnrolling, inferLogits, trainLogits, loss = tensorList
	return session, (input, output), [batchSize, outputLengthList, maximumUnrolling, (inferLogits, trainLogits)], [loss, trainingOp]

def testRun(args, sessionTuple, dictTuple):
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, logits = configTuple
	
	# Try feeding dummy data
	dummyInput = [[2, 6, 7, 9, 0], [4, 1, 1, 0, 0]]
	dummyOutput = [[9, 7, 6, 2, 0], [1, 4, 4, 0, 0]]
	dummyDecoderInput = [[5, 9, 7, 6, 2], [5, 1, 4, 4, 0]]
	dummyOutputLengthList = [4, 3]
	dummyMaximumUnrollingInBatch = 5
	dummyBatchSize = 2
	feed_dict = {input:dummyInput, output:dummyOutput, outputLengthList:dummyOutputLengthList, batchSize:dummyBatchSize, maximumUnrolling:dummyMaximumUnrollingInBatch, decoderInput:dummyDecoderInput}
	for i in range(1000):
		loss, _ = session.run(trainTuple, feed_dict=feed_dict)
	print("Latest loss: ", loss)
	testResultInfer, testResultTrain = session.run(logits, feed_dict=feed_dict)
	_, _, tgtEmbeddingVector = dictTuple[1]
	testResultInfer, _ = getWordIdFromVectors(testResultInfer[0], tgtEmbeddingVector, True)
	testResultTrain, _ = getWordIdFromVectors(testResultTrain[0], tgtEmbeddingVector, True)
	print("Correct output:\n", dummyOutput[0])
	print("Infer output:\n", testResultInfer)
	print("Train output:\n", testResultTrain)
	# print(session.run(logits[0], feed_dict={input:dummyInput, output:dummyOutput, outputLengthList:dummyOutputLengthList, batchSize:dummyBatchSize, maximumUnrolling:dummyMaximumUnrollingInBatch}))
	sys.exit()

	
def testSavingSession(args, sessionTuple):
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, logits = configTuple
	# assume session is constructed, run a dummy set
	dummyInput = [[2, 4, 5, 0], [6, 7, 2, 0]]
	dummyOutput = [[2, 4, 5, 0], [6, 7, 2, 0]]
	dummyOutputLengthList = [3, 3]
	dummyMaximumUnrollingInBatch = 4
	dummyBatchSize = 2
	feed_dict = {input:dummyInput, output:dummyOutput, outputLengthList:dummyOutputLengthList, batchSize:dummyBatchSize, maximumUnrolling:dummyMaximumUnrollingInBatch}
	savedLogit, _ = session.run(logits[0], feed_dict=feed_dict)
	# execute saving 
	createSessionStorageOps(session, sessionTupleToList(sessionTuple))
	builder.saveToPath(session, savePath)
	# delete the entire graph
	# reload
	builder.loadFromPath(session, savePath)
	sessionTuple = listToSessionTuple(getOpsFromSessionStorage(session))
	# run the dummy set again, this time with the loaded tensor/ops
	_, inputOutputTuple, configTuple, trainTuple = sessionTuple
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, logits = configTuple
	# run the dummy set again, this time with the loaded tensor/ops
	feed_dict = {input:dummyInput, output:dummyOutput, outputLengthList:dummyOutputLengthList, batchSize:dummyBatchSize, maximumUnrolling:dummyMaximumUnrollingInBatch}
	loadedLogit, _ = session.run(logits[0], feed_dict=feed_dict)
	print(savedLogit, loadedLogit)
	assert savedLogit[0][0] == loadedLogit[0][0]
	return session, inputOutputTuple, configTuple, trainTuple
	sys.exit()
	
def calculateBleu(correct, result, trimData=None):
	# calculate the bleu score using correct as baseline
	assert len(correct) == len(result)
	processedCorrect, processedResult = [], []
	for i in range(len(correct)):
		source, target = correct[i], result[i]
		# print(source, target)
		if(trimData is not None):
			correctLen = trimData[i]
			source = source[:correctLen]
			target = target[:correctLen]
		# leave the id as is, joining
		# may need to call the pieces instead
		processedCorrect.append(' '.join(map(str, source)))
		processedResult.append(' '.join(map(str, target)))
	return BLEU(processedResult, [processedCorrect])
	
if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Create training examples from resource data.')
	args = parser.parse_args()
	args.mode = 'train'
	args.directory = 'data\\vietchina'
	args.src_dict_file = 'vi_tokenized.embedding.bin'
	args.tgt_dict_file = 'ch_tokenized.embedding.bin'
	args.src_file = 'vi_tokenized.txt'
	args.tgt_file = 'ch_tokenized.txt'
	args.unknown_word = '*UNKNOWN*'
	args.end_token = '<\s>'
	args.start_token = '<s>'
	args.save = True
	args.save_path = 'save\\dummy'
	args.epoch = 100
	args.evaluation_step = 10
	args.global_steps = 0
	args.import_default_dict = True
	args.maximum_sentence_length = 50
	args.batch_size = 512
	timer = time.time()
	def getTimer():
		return time.time()-timer
	tf.reset_default_graph()
	embeddingTuple = createEmbeddingCouplingFromFile(args)
	sessionTuple = createSession(args, embeddingTuple)
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	if(args.save):
		savePath = os.path.join(args.directory, args.save_path + ".ckpt")
		builder.loadFromPath(session, savePath)
	print("Creating session done, time passed %.2fs" % getTimer())
	# testRun(args, sessionTuple, embeddingTuple)
	
	if(args.save and (not args.src_file or not args.tgt_file)):
		# try getting batches saved from previous iteration instead of creating new
		batchesFile = io.open(os.path.join(args.directory, args.save_path + ".bat"), 'rb')
		batches = pickle.load(batchesFile)
		batchesFile.close()
	elif(args.src_file and args.tgt_file):
		# create new batches from the files specified
		batchesCoupling = createSentenceCouplingFromFile(args)
		batches = generateBatchesFromSentences(args, batchesCoupling, embeddingTuple)
	else:
		raise argparse.ArgumentTypeError("Have no saved batches and no new src/tgt files for training. Exiting.")
	print("Batches generated/loaded, time passed %.2f, amount of batches %d" % (getTimer(), len(batches)))
	
	if(args.mode == 'train'):
		sample = generateRandomBatchesFromSet(args, batches, (embeddingTuple[0][0][args.end_token], embeddingTuple[1][0][args.end_token], embeddingTuple[1][0][args.start_token]))
		def evaluationFunction(extraArgs):
			iteration, losses = extraArgs
			trainResult = evaluateSession(args, sessionTuple, embeddingTuple, sample)
			_, correctOutput, _, trimLength, trainInput = sample
			print(trainInput[0], '\n=>', trainResult[0], '\n=', correctOutput[0])
			print(trainInput[6], '\n=>', trainResult[6], '\n=', correctOutput[6])
			# inferResult = calculateBleu(correctOutput, inferResult, trimLength)
			trainResult = calculateBleu(correctOutput, trainResult, trimLength)
			print("Iteration %d, time passed %.2f, BLEU score %2.2f" % (iteration, getTimer(), trainResult * 100.0))
			print("Losses during this cycle: {}".format(losses[-args.evaluation_step:]))
			# The evaluation decide which model should we be improving
			return
		# execute training
		evaluationFunction((0, []))
		totalLossTrack = trainSession(args, sessionTuple, batches, evaluationFunction)
	elif(args.mode == 'infer'):
		sample = generateRandomBatchesFromSet(args, batches, (embeddingTuple[0][0][args.end_token], embeddingTuple[1][0][args.end_token], embeddingTuple[1][0][args.start_token]))
		inferResult = evaluateSession(args, sessionTuple, embeddingTuple, sample)
		_, correctOutput, _, trimLength, trainInput = sample
		print(trainInput[0], '\n=>', trainResult[0], '\n=', correctOutput[0])
		print(trainInput[6], '\n=>', trainResult[6], '\n=', correctOutput[6])
		inferResult = calculateBleu(correctOutput, inferResult, trimLength)
		print("Inference mode ran, BLEU score %2.2f" % (inferResult * 100.0))
	else:
		raise argparse.ArgumentTypeError("Mode not registered. Please recheck.")
	if(args.save):
		builder.saveToPath(session, savePath)
		if(args.src_file and args.tgt_file and batches):
			batchesFile = io.open(os.path.join(args.directory, args.save_path + ".bat"), 'wb')
			pickle.dump(batches, batchesFile)
			batchesFile.close()
	print("All completed, total time passed %.2f" % getTimer())