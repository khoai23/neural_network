import ffBuilder as builder
import numpy as np
import tensorflow as tf
import sys, os, pickle, argparse, io, time, random, json
from calculatebleu import *

def getVocabFromVocabFile(fileDir):
	file = io.open(fileDir, 'r', encoding='utf-8')
	listVocab = [word.strip("\n ") for word in file.readlines()]
	file.close()
	return listVocab

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
			result.append(line.strip().split(splitToken))
		else:
			result.append([line.strip()])
	return result
	
def createSentenceCouplingFromFile(args, keepSentencesIndexes=False):
	srcSentences = getSentencesFromFile(os.path.join(args.directory, args.src_file))
	tgtSentences = getSentencesFromFile(os.path.join(args.directory, args.tgt_file))
	assert len(srcSentences) == len(tgtSentences)
	args.print_verbose('Sentences read from training files: %d' % len(srcSentences))
	coupling = []
	for i in range(len(srcSentences)):
		# filter out those which are too long
		if(len(srcSentences[i]) <= args.maximum_sentence_length and len(tgtSentences[i]) <= args.maximum_sentence_length):
			coupling.append((srcSentences[i], tgtSentences[i]))
		elif(args.trim_training_data):
			coupling.append((srcSentences[i][:args.maximum_sentence_length], tgtSentences[i][:args.maximum_sentence_length]))
	args.print_verbose('Sentences accepted from training files: %d' % len(coupling))
	# sentences loss only without args.trim_training_data
	assert (len(srcSentences) == len(coupling)) or not args.trim_training_data
	# Sort by number of words in output sentences
	if(keepSentencesIndexes):
		# this is an option generally for infer: batches must have indexes to be reordered after inference
		indexedCoupling = enumerate(coupling)
		indexedCoupling = sorted(indexedCoupling, key=lambda couple: (len(couple[1][1]), len(couple[1][0])))
		correctIndex, coupling = zip(*indexedCoupling)
	else:
		coupling = sorted(coupling, key=lambda couple: (len(couple[1]), len(couple[0])))
		correctIndex = None

	if(args.dev_file_name):
		# Try to get testing values
		srcDev = args.src + args.dev_file_name if(args.prefix) else args.dev_file_name + '.' + args.src
		srcDev = getSentencesFromFile(os.path.join(args.directory, srcDev))
		tgtDev = args.tgt + args.dev_file_name if(args.prefix) else args.dev_file_name + '.' + args.tgt
		tgtDev = getSentencesFromFile(os.path.join(args.directory, tgtDev))
		assert len(srcDev) == len(tgtDev)
		args.print_verbose('Sentences read from dev files: %d' % len(srcDev))
		otherCoupling = []
		
		for i in range(len(srcDev)):
			if(len(srcDev[i]) <= args.maximum_sentence_length and len(tgtDev[i]) <= args.maximum_sentence_length):
				otherCoupling.append((srcDev[i], tgtDev[i]))
				
		args.print_verbose('Sentences accepted from dev files: %d' % len(otherCoupling))
	else:
		otherCoupling = None
	return coupling, otherCoupling, correctIndex

	
def createEmbeddingCouplingFromFile(args):
	srcDict = getEmbeddingFromFile(os.path.join(args.directory, args.src_dict_file), args.import_default_dict)
	tgtDict = getEmbeddingFromFile(os.path.join(args.directory, args.tgt_dict_file), args.import_default_dict)
	# TODO Print a warning here for normal dict, since it may change order each time it is used. Then again, we are using OrderedDict and python3 stated that it preserve the sequence of words. So low priority.
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

def createCouplingFromVocabFile(args):
	srcWord = getVocabFromVocabFile(os.path.join(args.directory, args.src_dict_file))
	tgtWord = getVocabFromVocabFile(os.path.join(args.directory, args.tgt_dict_file))
	# check for <s> and <\s> in the vocab, and add if not existing. Start token might not need existing in srcWord, due to encoder not using it
	if(args.end_token not in srcWord):
		srcWord.append(args.end_token)
	if(args.start_token not in srcWord):
		tgtWord.append(args.start_token)
	if(args.end_token not in tgtWord):
		tgtWord.append(args.end_token)
	# unknownWord must be at the start of both vocab
	if(args.unknown_word in srcWord):
		srcWord.remove(args.unknown_word)
	if(args.unknown_word in tgtWord):
		tgtWord.remove(args.unknown_word)
	srcWord.insert(0, args.unknown_word)
	tgtWord.insert(0, args.unknown_word)
	# create the src/tgt list into normal and ref, embeddingVector will be None
	# initializer = np.random.normal if(args.vocab_init in ['gaussian', 'normal', 'xavier']) else np.random.uniform
	
	counter = 0
	srcWordToId, srcIdToWord = {}, {}
	for key in srcWord:
		srcWordToId[key] = counter
		srcIdToWord[counter] = key
		counter += 1
	
	counter = 0
	tgtWordToId, tgtIdToWord = {}, {}
	tgtEmbeddingVector = []
	for key in tgtWord:
		tgtWordToId[key] = counter
		tgtIdToWord[counter] = key
		counter += 1
		
	
	return (srcWordToId, srcIdToWord, None), (tgtWordToId, tgtIdToWord, None)
	
def createSession(args, embedding):
	srcEmbedding, tgtEmbedding = embedding
	srcEmbeddingDict, _, srcEmbeddingVector = srcEmbedding
	tgtEmbeddingDict, _, tgtEmbeddingVector = tgtEmbedding
	if(srcEmbeddingVector is None or tgtEmbeddingVector is None):
		# initialize in vocab mode - random due to arguments
		embeddingSize = args.layer_size
		tgtNumWords = len(tgtEmbeddingDict)
		minVal, maxVal = args.initialize_range
		if(args.vocab_init in ['gaussian', 'normal', 'xavier']):
			srcEmbeddingVector = tf.random_normal([len(srcEmbeddingDict), embeddingSize], mean=(maxVal+minVal)/2, stddev=(maxVal-minVal)/2)
			tgtEmbeddingVector = tf.random_normal([len(tgtEmbeddingDict), embeddingSize], mean=(maxVal+minVal)/2, stddev=(maxVal-minVal)/2)
		else:
			srcEmbeddingVector = tf.random_uniform([len(srcEmbeddingDict), embeddingSize], minval=minVal, maxval=maxVal)
			tgtEmbeddingVector = tf.random_uniform([len(tgtEmbeddingDict), embeddingSize], minval=minVal, maxval=maxVal)
	else:
		# srcEmbeddingSize = srcEmbeddingVector.shape[1]
		embeddingSize = tgtEmbeddingVector.shape[1]
		tgtNumWords = tgtEmbeddingVector.shape[0]
		assert tgtNumWords == len(tgtEmbeddingDict)
	
	srcEmbeddingVector = tf.Variable(srcEmbeddingVector, dtype=tf.float32, trainable=args.train_embedding, name='input_embedding')
	tgtEmbeddingVector = tf.Variable(tgtEmbeddingVector, dtype=tf.float32, trainable=args.train_embedding, name='output_mbedding')
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	# set the initializer to the entire session according to args.vocab_init
	minVal, maxVal = args.initialize_range
	initializer = tf.random_normal_initializer(mean=(maxVal+minVal)/2, stddev=(maxVal-minVal)/2) if(args.vocab_init in ['gaussian', 'normal', 'xavier']) \
			else  tf.random_uniform_initializer(minval=minVal, maxval=maxVal)
	args.print_verbose("Initializer range (%.2f -> %.2f), type %s" % (minVal, maxVal, args.vocab_init))
	
	tf.get_variable_scope().set_initializer(initializer)
	# dropout value, used for training. Must reset to 1.0(all) when infer
	dropout = tf.placeholder_with_default(1.0, shape=(), name='dropout')
	# input in shape (batchSize, inputSize) - not using timemayor
	input = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input')
	# input are lookup from the known srcEmbeddingVector, shape (batchSize, inputSize, embeddingSize)
	inputVector = tf.nn.embedding_lookup(srcEmbeddingVector, input, name='input_encoder_vectors')
	# craft the encoder depend on the input vector. Currently using default values for all version
	settingDict = {'inputType':inputVector, 'layerSize':embeddingSize, 'layerDepth':args.layer_depth, 'inputSize':None, 'dropout':dropout, 'bidirectional':True}
	inputFromEncoder, encoderOutput, encoderState, dropoutFromEncoder = builder.createEncoder(settingDict)
	assert inputFromEncoder is inputVector and dropoutFromEncoder is dropout
	# craft the output in shape (batchSize, outputSize)
	output = tf.placeholder(shape=[None, None], dtype=tf.int32, name='output')
	decoderInput = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input_decoder')
	# the inputLengthList is the length of the encoding sentence, necessary for attention to accurately select within the correct input sentence
	inputLengthList = tf.placeholder(shape=[None], dtype=tf.int32, name='input_length')
	# the outputLengthList is the length of the sentence supposed to be output. Used to create somewhat more accurate loss function
	outputLengthList = tf.placeholder(shape=[None], dtype=tf.int32, name='output_length')
	# These are the dimension of the batch in decoder. Needed for retarded high-level decoder functions.
	batchSize = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')
	maximumUnrolling = tf.placeholder_with_default(args.maximum_sentence_length, shape=(), name='decoder_maximum_length')
	# likewise, the output will be looked up into shape (batchSize, inputSize, embeddingSize)
	# outputVector = tf.nn.embedding_lookup(tgtEmbeddingVector, output)
	# stop using decoderInputVector as a test
	# decoderInputVector = tf.nn.embedding_lookup(tgtEmbeddingVector, decoderInput, name='input_decoder_vectors')
	# decoder will use the encoderState to work, outputVector and tgtEmbeddingVector for lookup check
	# also need a mode placeholder for switching between decoder helper and the start/end tokenId to search for 
	startTokenId, endTokenId = tgtEmbeddingDict[args.start_token], tgtEmbeddingDict[args.end_token]
	args.print_verbose("Start token is %d. End token is %d." % (startTokenId, endTokenId))
	# mode = tf.placeholder_with_default(True, shape=())
	# construct the settingDict
	# the following settings is MANDATORY. Without any of the below setting, it simply won't work
	settingDict['startTokenId'] = startTokenId; settingDict['endTokenId'] = endTokenId
	settingDict['correctResult'] = output; settingDict['outputEmbedding'] = tgtEmbeddingVector;
	settingDict['correctResultLen'] = outputLengthList; settingDict['encoderState'] = encoderState; settingDict['decoderOutputSize'] = tgtNumWords
	settingDict['batchSize'] = batchSize; settingDict['maximumDecoderLength'] = maximumUnrolling; # settingDict['decoderInput'] = decoderInputVector
	if(args.attention):
		# Duplicate spotted
		# inputLengthList = tf.placeholder(shape=[None], dtype=tf.int32, name='')
		settingDict['attention'] = args.attention
		settingDict['encoderOutput'] = encoderOutput
		settingDict['encoderLengthList'] = inputLengthList
	else:
		# as feed_dict do not allow None in 1.5, we leave the placeholder be
		# inputLengthList = None
		pass
	
	settingDict['globalSteps'] = tf.train.get_or_create_global_step() #tf.Variable(args.global_steps, trainable=False, dtype=tf.int32, name='global_steps')
	if(args.scheduled_sampling_rate > 0.0 and args.scheduled_sampling_step > 0):
		# Create the needed steps for the decoder to use
		stairStep = tf.to_float(settingDict['globalSteps'] // tf.constant(args.scheduled_sampling_step, dtype=settingDict['globalSteps'].dtype))
		if(args.scheduled_sampling_type == 'linear'):
			settingDict['samplingVariable'] = tf.maximum(1.0 - args.scheduled_sampling_rate * stairStep, 1.0)
		elif(args.scheduled_sampling_type == 'exp'):
			settingDict['samplingVariable'] = tf.pow(args.scheduled_sampling_rate, stairStep)
		elif(args.scheduled_sampling_type == 'inv_sigmoid'):
			k = args.scheduled_sampling_rate
			settingDict['samplingVariable'] = k / (k + tf.exp(stairStep / k))
	
	logits, loss, outputIds, _ = builder.createDecoder(settingDict)
	# TrainingOp function, built on the loss function
	settingDict['mode'] = args.optimizer
	settingDict['trainingRate'] = args.learning_rate
	settingDict['incrementGlobalStep'] = tf.assign_add(settingDict['globalSteps'], 1)
	
	if(args.dynamic_clipping):
		args.dynamic_clipping = tf.placeholder(shape=(), dtype=tf.float32)
		args.gradient_clipping = args.dynamic_clipping * tf.constant(args.gradient_clipping)
	
	if(True):
		# Manual construction of training op, currently not in use
		if(args.warmup_threshold > 0):
			args.print_verbose("Has warmup in setting.")
			if(args.warmup_steps <= 0):
				args.warmup_steps = args.warmup_threshold // 5
			settingDict['warmupTraining'] = (args.warmup_steps, args.warmup_threshold)
		if(args.decay_threshold >= 0):
			args.print_verbose("Has decay in setting.")
			settingDict['decayTraining'] = (args.decay_steps, args.decay_threshold, args.decay_factor)
			settingDict['disableStaircase'] = args.decay_disable_staircase
			
		settingDict['loss'] = loss
		trainingTrainOp = builder.createOptimizer(settingDict)
		# All ops will return (optimizer, incrementGlobalStep) tuple, the second one only available in sgd warmup/decay
		settingDict['colocateGradient'] = args.colocate
		settingDict['clipGradient'] = args.gradient_clipping
		trainingGradient = builder.configureGradientOptions(trainingTrainOp, settingDict)
		trainingTrainOp = tf.group(trainingGradient, settingDict['incrementGlobalStep'])
	else:
		# optimizerName = 'SGD' if args.optimizer.lower()=='sgd' else 'Adam' if args.optimizer.lower()=='adam' else None
		if(args.decay_threshold >= 0):
			threshold = tf.constant(args.decay_threshold)
			def decayFunction(trainingRate, globalSteps):
				return tf.cond(global_steps >= threshold, 
							true_fn=lambda: exponential_decay(trainingRate, tf.maximum(globalSteps - threshold, 0), args.decay_steps, args.decay_factor, staircase=True),
							false_fn=trainingRate)
		else:
			decayFunction = None
		
		trainingTrainOp = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), args.learning_rate, args.optimizer,
				clip_gradients=args.gradient_clipping, learning_rate_decay_fn=decayFunction, colocate_gradients_with_ops=args.colocate, name='optimizer')
	# initiate the session
	session.run(tf.global_variables_initializer())
	
	if(args.verbose):
		for key in settingDict:
			args.print_verbose("{}:{}".format(key, type(settingDict[key]) if(not isinstance(settingDict[key], (int, str, bool, dict, list, tuple))) else settingDict[key]))
	
	inputOutputTuple = [input, output, decoderInput]
	configTuple = [inputLengthList, outputLengthList, batchSize, maximumUnrolling, dropout, outputIds]
	trainTuple = [loss, trainingTrainOp]
	
	return session, inputOutputTuple, configTuple, trainTuple
	
def trainSession(args, sessionTuple, batches, evaluationFunction=None):
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	input, output, decoderInput = inputOutputTuple
	inputLengthList, outputLengthList, batchSize, maximumUnrolling, dropout, _ = configTuple
	avgLosses = [0]
	loss = 1.0
	bestValue = -1.0
	for step in range(args.epoch):
		#if(not args.train_greedy):
		#	args.print_verbose(("Use TrainingHelper in iteration %d" if(useTrainingHelper) else "Use GreedyEmbeddingHelper in iteration %d") % step)
		for batch in batches:
			args.global_steps += 1
			trainInput, trainCorrectOutput, trainInputLengthList, trainOutputLengthList, trainDecoderInput = batch
			feed_dict = {input:trainInput, output:trainCorrectOutput, decoderInput:trainDecoderInput, inputLengthList:trainInputLengthList, outputLengthList:trainOutputLengthList, \
				batchSize:len(trainInput), maximumUnrolling:max(trainOutputLengthList), dropout:args.dropout}
			if(args.dynamic_clipping is not False):
				feed_dict[args.dynamic_clipping] = loss
			loss, _ = session.run(trainTuple, feed_dict=feed_dict)
			if(np.isnan(loss)):
				print("Loss nan @ global_steps {}, feed_dict {}".format(args.global_steps, feed_dict))
				findNanSession(args, session)
				sys.exit(0)
			#else:
			#	args.print_verbose("Loss %.4f @ global_steps %d" % (loss, args.global_steps))
			avgLosses[-1] += loss
			if(args.verbose):
				if(args.global_steps % args.verbose_timer == 0):
					args.print_verbose("Global step %d, last loss on batch %2.4f, time passed %.2f" % (args.global_steps, loss, args.time_passed()))
			if(args.global_steps % args.debug_steps == 0 and args.debug):
				debugSession(args, session)
		avgLosses[-1] = avgLosses[-1] / len(batches)
		if(evaluationFunction and (step+1) % args.evaluation_step == 0):
			# run evaluationFunction every evaluation_step epoch
			bestValue = evaluationFunction((step+1,avgLosses, bestValue))
		avgLosses.append(0)
	return avgLosses
	
def evaluateSession(args, sessionTuple, dictTuple, sampleBatch):
	session, inputOutputTuple, configTuple, _ = sessionTuple
	input, output, decoderInput = inputOutputTuple
	inputLengthList, outputLengthList, batchSize, maximumUnrolling, dropout, outputIds = configTuple
	_, _, tgtEmbeddingVector = dictTuple[1]
	sampleInput, sampleCorrectOutput, sampleInputLengthList, sampleOutputLengthList, sampleDecoderInput = sampleBatch
	# feed_dict = {input:sampleInput, outputLengthList:sampleOutputLengthList, batchSize:sampleBatch[2], maximumUnrolling:max(sampleBatch[3]), decoderInput:sampleDecoderInput, dropout:1.0}
	feed_dict = { input:sampleInput, output:sampleCorrectOutput, inputLengthList:sampleInputLengthList, outputLengthList:sampleOutputLengthList, \
				batchSize:len(sampleInput), maximumUnrolling:max(sampleOutputLengthList), dropout:1.0 }
	# print(feed_dict.keys())
	sampleResult = session.run(outputIds, feed_dict=feed_dict)
	return sampleResult

def inferenceSession(args, session, data):
	session, inputOutputTuple, configTuple, _ = sessionTuple
	input, _, _ = inputOutputTuple
	inputLengthList, _, batchSize, maximumUnrolling, dropout, outputIds = configTuple
	_, inferrenceOutput = outputIds
	# Use default values for maximumUnrolling. Defaulted, but just do it to be sure. outputLengthList and decoderInput should not be neccessary as we are calling only GreedyEmbeddingHelper
	output = []
	for infInput, infInputLength in data:
		feed_dict = {input:infInput, inputLengthList:infInputLength, batchSize:len(infInput), maximumUnrolling:args.maximum_sentence_length, dropout:1.0}
		decodeOutput = session.run(inferrenceOutput, feed_dict=feed_dict)
		output.append(decodeOutput)
	return output
	
def findNanSession(args, session):
	# Search for all variable within the session
	varList = tf.trainable_variables()
	for tensor in varList:
		tensorValue = tensor.eval(session=session)
		# use dimSize to determine if int or ndarray, flatten if ndarray
		if(not isinstance(tensorValue, (np.ndarray, list))):
			if(not np.isfinite(tensorValue)):
				printNan(args, tensor.name)
		else:
			if(not all(np.isfinite(val) for val in tensorValue.flatten())):
				printNan(args, tensor.name)

def printNan(args, tensorName):
	print("Tensor %s had nan in its values" % tensorName)

def debugSession(args, session):
	# Get all independent variables in the sessions
	tensorList = tf.trainable_variables()
	# Maximum and minimum values, generate abitrary number to be overwritten
	maxVal, minVal, maxValTensor, minValTensor = -1e20, 1e20, None, None
	# Values taken from the session
	tensorValues = session.run(tensorList)
	for tensor, values in zip(tensorList, tensorValues):
		name = tensor.name
		if(isinstance(values, np.ndarray)):
			# flatten
			values = values.flatten()
		else:
			# create single length values
			values = np.array([values])
		args.print_verbose("Tensor %s, shape when flattened %s" % (name, values.shape))
		assert len(values.shape) == 1
		# iterate, record
		for v in values:
			if(v > maxVal):
				maxVal, maxValTensor = v, name
			elif(v < minVal):
				minVal, minValTensor = v, name
	args.print_verbose("Maximum @%s:%.2f; Minimum @%s:%.2f." % (maxValTensor, maxVal, minValTensor, minVal))
	# Return the values
	return (maxVal, maxValTensor), (minVal, minValTensor)

def generateBatchesFromSentences(args, data, embeddingTuple, singleBatch=False):
	srcDictTuple, tgtDictTuple = embeddingTuple
	srcWordToId, tgtWordToId = srcDictTuple[0], tgtDictTuple[0]
	srcUnknownID, tgtUnknownID = srcWordToId[args.unknown_word], tgtWordToId[args.unknown_word]
	startTokenPad = [tgtWordToId[args.start_token]]
	# data are binding tuples of (s1, s2) for src-tgt, s1/s2 preprocessed into array of respective words
	batches = []
	srcBatch, tgtBatch = [], []
	for srcSentence, tgtSentence in data:
		srcSentence = [srcWordToId.get(word, srcUnknownID) for word in srcSentence]
		tgtSentence = [tgtWordToId.get(word, tgtUnknownID) for word in tgtSentence]
		srcBatch.append(srcSentence)
		tgtBatch.append(tgtSentence)
		if(len(srcBatch) == args.batch_size and not singleBatch):
			# Full batch, begin converting. If singleBatch, will not go here
			assert len(srcBatch) == len(tgtBatch)
			inputLengthList = padMatrix(srcBatch, srcWordToId[args.end_token])
			outputLengthList = padMatrix(tgtBatch, tgtWordToId[args.end_token])
			tgtInputBatch = [ (startTokenPad + list(tgt))[:-1] for tgt in tgtBatch]
			batches.append((srcBatch, tgtBatch, inputLengthList, outputLengthList, tgtInputBatch))
			srcBatch, tgtBatch = [], []
	# Last batch
	inputLengthList = padMatrix(srcBatch, srcWordToId[args.end_token])
	outputLengthList = padMatrix(tgtBatch, tgtWordToId[args.end_token])
	tgtInputBatch = [ (startTokenPad + list(tgt))[:-1] for tgt in tgtBatch]
	batches.append((srcBatch, tgtBatch, inputLengthList, outputLengthList, tgtInputBatch))
	# Return the processed value
	return batches
	
def checkBatchValidity(args, batch, embeddingTuple):
	# Check if all input/output values in batch is in the boundary of dictTuple 
	srcEmbeddingTuple, tgtEmbeddingTuple = embeddingTuple
	srcIdToWord, tgtIdToWord = srcEmbeddingTuple[1], tgtEmbeddingTuple[1]
	srcSize, tgtSize = len(srcIdToWord), len(tgtIdToWord)
	# args.print_verbose("Found srcSize: %d, tgtSize: %d" % (srcSize, tgtSize))
	# check on 0-1-4: input-output-tgtInput
	input, output, _, _, tgtInput = batch
	if(any(idx >= srcSize for idx in np.array(input).flatten())):
		raise ValueError("Caught invalid index @input, full input {} while maximumIdx {}".format(input, srcSize))
	if(any(idx >= tgtSize for idx in np.array(output).flatten())):
                raise ValueError("Caught invalid index @output, full output {} while maximumIdx {}".format(output, tgtSize))
	if(any(idx >= tgtSize for idx in np.array(tgtInput).flatten())):
                raise ValueError("Caught invalid index @tgtInput, fullInput {} while maximumIdx {}".format(tgtInput, tgtSize))

def generateRandomBatchesFromSet(args, batches, paddingToken):
	raise Exception("Unfixed @generateRandomBatchesFromSet")
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
	
def generateInferenceInputFromFile(args, embeddingTuple):
	# Will only concern src side and single file
	inferSentences = getSentencesFromFile(os.path.join(args.directory, args.src_file))
	srcDictTuple, _ = embeddingTuple
	srcWordToId = srcDictTuple[0]
	unknownWordId = srcWordToId[args.unknown_word]
	# Create input data
	return generateInferenceBatchesFromData(inferSentences, srcWordToId, unknownWordId)

def generateInferenceBatchesFromData(data, srcWordToId=None, unknownWordId=None):
	assert srcWordToId is not None and unknownWordId is not None
	inferSentences = [[srcWordToId.get(word, unknownWordId) for word in sentence] for sentence in data]
	# default will have indexes of sorted sentences
	indexedSentences = enumerate(inferSentences)
	indexedSentences = sorted(indexedSentences, key=lambda idx, x:len(x))
	indexes, inferSentences = zip(*indexedSentences)
	# Split into smaller batch to avoid overruning the memory with batches too large. 
	batchSize = args.batch_size
	numInferBatches = len(inferSentences) / batchSize + 1
	rawBatches = [inferSentences[i*batchSize:(i+1)*batchSize] for i in range(numInferBatches)]
	# the smaller batches will be zipped as input matrix and length
	formattedBatches = []
	for rawBatch in rawBatches:
		lengthList = padMatrix(rawBatch)
		formattedBatches.append((rawBatch, lengthList))
	return formattedBatches, indexes
		
def padMatrix(matrix, paddingToken):
	# find the longest line in the matrix to do the padding
	# TODO plus one for the longest if not crossing abitrary maxLength
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
	
def testRun(args, sessionTuple, dictTuple):
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	input, output, decoderInput = inputOutputTuple
	batchSize, outputLengthList, maximumUnrolling, logits, outputIds = configTuple
	
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
	
def outputInferenceToFile(args, embeddingTuple, inferOutput, reorderIdx=None, leftAsIdx=False):
	if(leftAsIdx):
		raise ValueError("leftAsIdx option no longer supported.")
	outputFilePath = args.tgt + args.output_file_name if(args.prefix) else args.output_file_name + '.' + args.tgt
	_, tgtDictTuple = embeddingTuple
	tgtWordToId, tgtIdToWord, _ = tgtDictTuple
	endTokenId = tgtWordToId[args.end_token]
	with io.open(os.path.join(args.directory, outputFilePath), 'w', encoding='utf-8') as outputFile:
		writeData = convertInferenceToWriteData(inferOutput, tgtIdToWord=tgtIdToWord, endTokenId=endTokenId, reorderIdx=reorderIdx)
		outputFile.writelines(writeData)
	return outputFilePath

def convertInferenceToWriteData(inferOutput, tgtIdToWord=None, endTokenId=None, separator=' ', reorderIdx=None):
	assert tgtWordToId is not None and endTokenId is not None
	if(reorderIdx):
		# reorder the inferOutput basing on the indexes if available
		reorderedSentences = sorted(zip(reorderIdx, inferOutput))
		_, inferOutput = zip(*reorderedSentences)
	formattedSentences = []
	for sentence in inferOutput:
		sentenceEnd = sentence.find(endTokenId)
		if(sentenceEnd >= 0):
			sentence = sentence[:sentenceEnd]
		sentence = [tgtIdToWord[int(wordIdx)] for wordIdx in sentence]
		sentence = separator.join(sentence)
		formattedSentences.append(sentence)
	return formattedSentences
	
def calculateBleu(correct, result, trimData=None):
	# calculate the bleu score using correct as baseline
	assert len(correct) == len(result)
	processedCorrect, processedResult = [], []
	for i in range(len(correct)):
		source, target = correct[i], result[i]
		# print(source, target)
		if(trimData is not None):
			correctLen = min(trimData[i], len(source), len(target))
			source = source[:correctLen]
			target = target[:correctLen]
		# leave the id as is, joining
		# may need to call the pieces instead
		processedCorrect.append(' '.join(map(str, source)))
		processedResult.append(' '.join(map(str, target)))
	return BLEU(processedResult, [processedCorrect])
	
	
def stripResultArray(sentence, token):
	# remove all token at the end of the sentence save one
	# print(token)
	if(isinstance(sentence, np.ndarray)):
		sentence = sentence.tolist()
	try:
		sentence = sentence[:sentence.index(token)]
	except ValueError:
		pass
	return sentence
	
def paramsSaveList(str):
	if(str == 'all'):
		return False, []
	elif(str[0] not in "ie"):
		raise argparse.ArgumentTypeError("Must be i/e @save_params")
	mode = str[0] == 'i'
	paramList = str.strip("[] ").split(" ,")
	return mode, paramList
	
def tryLoadOrSaveParams(args, exception=None):
	if(args.params_path is None):
		if(args.verbose):
			print("Params path not found, default to save_path.")
		args.params_path = os.path.join(args.directory, args.save_path)
	if(args.params_path and ".params" not in args.params_path):
		args.params_path += ".params"
	
	if(args.params_mode == 'pickle'):
		loaderFunc = pickle.load
		saverFunc = pickle.dump
	elif(args.params_mode == 'json'):
		loaderFunc = json.load
		saverFunc = json.dump
	else:
		raise ArgumentTypeError("Unrecognized params_mode {:s}, must be json/pickle.".format(args.params_mode))

	if(args.load_params):
		with io.open(args.params_path, 'rb') as paramFile:
			paramValues = loaderFunc(paramFile)
		for param in paramValues:
			setattr(args, param, paramValues[param])
	elif(args.save_params):
		include_save_mode, listParams = args.save_params
		if(include_save_mode):
			# Save all params taken in this mode
			listParams = [param for param in listParams if param in vars(args)]
		else:
			# Exclude all params in this mode
			listParams = [param for param in vars(args) if param not in listParams]
		if(isinstance(exception, list)):
			# Some params will be automatically removed regardless of options
			listParams = list(filter(lambda x: x in exception, listParams))
		if(len(listParams) == 0):
			raise argparse.ArgumentTypeError("Params list @save_params invalid.")
		dictParams = dict((param, getattr(args, param)) for param in listParams)
		with io.open(args.params_path, 'wb') as paramFile:
			saverFunc(dictParams, paramFile)
	
def strToRange(str):
	try:
		val = np.abs(float(str))
		return (-val, val)
	except ValueError:
		str = str.strip("() ").split(',|')
		return float(str[0], float(str[1]))
	
def constructParser(loadDefault=False):
		# Run argparse
	parser = argparse.ArgumentParser(description='Create training examples from resource data.')
	# OVERALL CONFIG
	parser.add_argument('-m','--mode', type=str, default='train', help='Mode to run the file. Currently only train|infer')
	parser.add_argument('--read_mode', type=str, default='embedding', help='Read binary, pickled, dictionary files as embedding, or vocab files as vocab. Default embedding')
	parser.add_argument('--import_default_dict', action='store_false', help='Do not use the varied length original embedding instead of the normalized version.')
	parser.add_argument('--train_embedding', action='store_true', help='Train the embedding vectors of words during the training. Will be forced to True in vocab read_mode.')
	parser.add_argument('--vocab_init', type=str, default='uniform', help='Choose type of initializer for vocab mode. Default uniform, can be normal(gaussian).')
	parser.add_argument('--initialize_range', type=strToRange, default=(-0.1, 0.1), help='The range to initialize variables, default (-0.1, 0.1).')
	parser.add_argument('--directory', type=str, default=os.path.curdir, help='The dictionary to keep all the training files. Default to the running directory.')
	parser.add_argument('-e', '--embedding_file_name', type=str, default='vocab', help='The names of the files for vocab or embedding. Default vocab.')
	parser.add_argument('-i', '--input_file_name', type=str, default='input', help='The names of the files for training or inference. Default input.')
	parser.add_argument('-o', '--output_file_name', type=str, default='output', help='The names of the files to write inference result in. Default output.')
	parser.add_argument('-d', '--dev_file_name', type=str, default=None, help='The names of the files for testing. If not specified, will take a random batch as the test intead.')
	parser.add_argument('-v', '--verbose', action='store_true', help='Print possibly pointless, particularly pitiful piece.')
	parser.add_argument('--tgt', type=str, required=True, help='The extension/prefix for the target files.')
	parser.add_argument('--src', type=str, required=True, help='The extension/prefix for the source files.')
	parser.add_argument('--prefix', action='store_true', help='If specified, tgt and src will be the prefix for the files.')
	parser.add_argument('--unknown_word', type=str, default='<unk>', help='Key in embedding/vocab standing for unknown word')
	parser.add_argument('--start_token', type=str, default='<s>', help='Key in embedding/vocab standing for the starting token')
	parser.add_argument('--end_token', type=str, default='<\s>', help='Key in embedding/vocab standing for the ending token')
	parser.add_argument('-s', '--save_path', type=str, default=None, help='Directory to load and save the trained model. Required in training mode.')
	parser.add_argument('-b', '--batch_file_name', type=str, default=None, help='The names of created batch file for training. If not specified, will try to look for it in save_path with "bat" extension .')
	parser.add_argument('--save_batch', action='store_true', help='If specified, the batch will be saved to batch_file_name for future training.')
	parser.add_argument('--shuffle_batch', action='store_true', help='If specified, the batch will be shuffled randomly for each training execution.')
	parser.add_argument('--epoch', type=int, default=100, help='How many times this model will train on current data. Default 100.')
	parser.add_argument('--maximum_sentence_length', type=int, default=50, help='Maximum length of sentences. Default 50.')
	parser.add_argument('--batch_size', type=int, default=128, help='Size of mini-batch for training. Default 128.')
	
	parser.add_argument('--layer_size', type=int, default=128, help='Size of hidden layer in each cell. Default to 128. Will be rewritten to size of embedding if in embedding read_mode.')
	parser.add_argument('--layer_depth', type=int, default=2, help='Depth of the greatest cell (number of sub-cell within it). Default 2.')
	parser.add_argument('--beam_width', type=int, default=1, help='The width of the beam to be used in BeamSearchDecoder, with 1 using Greedy.')
	
	# SAVE STUFF
	parser.add_argument('--load_params', action='store_true', help='Use a json or pickle file as setting. All arguments found in file will be overwritten.')
	parser.add_argument('--save_params', type=paramsSaveList, default=None, help='Save the current params, use i[] or e[] for including(only save the specified) or excluding(save all but the specified)')
	parser.add_argument('--params_mode', type=str, default='pickle', help='Pickle or JSON. JSON not implemented in the near future')
	parser.add_argument('-p', '--params_path', type=str, default=None, help='The path of params. If not specified, take save_path as subtitution. Will cause exception as normal in load_params.')
	
	# MODEL CONFIG
	parser.add_argument('-a', '--attention', type=str, default='scaled_luong', help='Use attention architecture. If not bahdanau/luong, will not use attention')
#	parser.add_argument('--train_greedy', action='store_true', help='If specified, will attempt to train using the GreedyEmbeddingHelper when its accuracy is worse than TrainingHelper.')
	parser.add_argument('--colocate', action='store_true', help='If specified, do colocate regarding gradient calculation.')
	parser.add_argument('--dropout', type=float, default=1.0, help='The dropout used for training. Will be automatically set to 1.0 in infer mode. Default 1.0')
	parser.add_argument('--optimizer', type=str, default='SGD', help='The optimizer used for training. List in tf.contrib.layer.optimize_loss function . Default SGD.')
	parser.add_argument('--learning_rate', type=float, default=None, help='The learning rate used for training. Default 1.0 for SGD and 0.001 for Adam.')
	parser.add_argument('--warmup_threshold', type=int, default=0, help='The warmup step used for learning rate (on global steps). If unspecified, will not use warmup.')
	parser.add_argument('--warmup_steps', type=int, default=0, help='The warmup factor for steps. Default 1/5 of the threshold.')
	parser.add_argument('--decay_threshold', type=int, default=-1, help='The threshold to begin decay. If unspecified, will not use decay.')
	parser.add_argument('--decay_steps', type=int, default=1000, help='The steps to staircase the learning rate decay (on global steps). Default 1000.')
	parser.add_argument('--decay_disable_staircase', action='store_true', help='If used, will use staircase decay. It doesnt impact performance, so a secondary option.')
	parser.add_argument('--decay_factor', type=float, default=0.5, help='The factor to multiply at each decay_steps. Default 0.5')
	parser.add_argument('--gradient_clipping', type=float, default=5.0, help='The maximum value for gradient. Default 5.0')
	parser.add_argument('--dynamic_clipping', action='store_true', help='If activate, clip the gradients based on the losses multiplying the gradient_clipping variable.')
	parser.add_argument('--scheduled_sampling_rate', type=float, default=0.0, help='If specified > 0.0, use ScheduledEmbeddingTrainingHelper with the rate with step')
	parser.add_argument('--scheduled_sampling_step', type=int, default=0, help='If specified a positive integer, use ScheduledEmbeddingTrainingHelper with the step specified')
	parser.add_argument('--scheduled_sampling_type', type=str, default='linear', help='Use linear|exp|inv_sigmoid in the ScheduledEmbeddingTrainingHelper. Default use linear')
	parser.add_argument('--save_when_evaluate', action='store_true', help='If specfied, save a .best model in the default save path')
	parser.add_argument('--trim_training_data', action='store_true', help='If specfied, sentences with oversized length are trimmed instead of dropped')
	# DEBUG
	parser.add_argument('--debug', action='store_true', help='When activated, run debugSession function every debug_steps during training.')
	parser.add_argument('--debug_steps', type=int, default=1, help='The step to run debug function. Default to every step (1).')
	parser.add_argument('--evaluation_step', type=int, default=2, help='For each evaluation_step epoch, run the check for accuracy. Default 2.')
	parser.add_argument('--verbose_timer', type=int, default=100, help='The global step to run timer. Default to 100.')

	if(loadDefault):
		args = parser.parse_args([])
	else:
		args = parser.parse_args()
	return args
	
def constructAndLoadSession(args):
	tf.reset_default_graph()
	if(args.read_mode == 'embedding'):
		embeddingTuple = createEmbeddingCouplingFromFile(args)
	elif(args.read_mode == 'vocab'):
		embeddingTuple = createCouplingFromVocabFile(args)
	sessionTuple = createSession(args, embeddingTuple)
	session = sessionTuple[0]
	if(args.save_path):
		savePath = os.path.join(args.directory, args.save_path + ".ckpt")
		builder.loadFromPath(session, savePath)
	print("Creating session done, time passed %.2fs" % getTimer())
	return sessionTuple, embeddingTuple

def saveSession(args, sessionTuple):
	session = sessionTuple[0]
	if(args.save_path):
		savePath = os.path.join(args.directory, args.save_path + ".ckpt")
		builder.saveToPath(session, savePath)
		if(batches and args.save_batch):
			if(not args.batch_file_name):
				args.batch_file_name = os.path.join(args.directory, args.save_path + ".bat")
			batchesFile = io.open(args.batch_file_name, 'wb')
			pickle.dump((batches, sample), batchesFile)
			batchesFile.close()

def loadDataByMode(args, embeddingTuple):
	# refer to creation of embeddingTuple for why
	srcWordToId, tgtWordToId = embeddingTuple[0][0], embeddingTuple[1][0]
	if(args.mode == 'train'):
		# In train, must be a batch of input/output matrix and respective lengths
		otherBatchFileName = os.path.join(args.directory, args.save_path + ".bat")
		if(os.path.isfile(otherBatchFileName) or (args.batch_file_name and os.path.isfile(args.batch_file_name))):
			# try getting batches saved from previous iteration instead of creating new
			batchesPath = args.batch_file_name if(not os.path.isfile(otherBatchFileName)) else otherBatchFileName
			args.print_verbose("Load batch data from pickle at {:s}".format(batchesPath))
			with io.open(batchesPath, 'rb') as batchesFile:
				batches, sample = pickle.load(batchesFile)
		else:
			# If cannot find the data, create from files in direction
			#try:
				# create new batches from the files src_file and tgt_file specified
				args.print_verbose("Load batch data from default input directory.")
				batchesCoupling, sampleCoupling, _ = createSentenceCouplingFromFile(args)
				batches = generateBatchesFromSentences(args, batchesCoupling, embeddingTuple)
				if(sampleCoupling is not None):
					sample = generateBatchesFromSentences(args, sampleCoupling, embeddingTuple)[0]
				else:
					sample = generateRandomBatchesFromSet(args, batches, (srcWordToId[args.end_token], tgtWordToId[args.end_token], tgtWordToId[args.start_token]))
			#except Exception:
			#	raise argparse.ArgumentTypeError("Have no saved batches and no new src/tgt files for training. Exiting.")
		# Check batch for screwup in idx values
		for batch in batches:
			checkBatchValidity(args, batch, embeddingTuple)	
		print("Batches generated/loaded with no error, time passed %.2f, amount of batches %d" % (getTimer(), len(batches)))
		args.print_verbose("Size of sampleBatch: %d" % len(sample[2]))
		if(args.shuffle_batch):
			random.shuffle(batches)
		return batches, sample
	elif(args.mode == 'infer'):
		# In infer, can be batch of input/output if tgt detected; else only the inputs and its lengths
		if(os.path.isfile(os.path.join(args.directory, args.tgt_file))):
			# Has coupling possible, use default functions
			batchesCoupling, _, correctIndex = createSentenceCouplingFromFile(args, keepSentencesIndexes=True)
			batches = generateBatchesFromSentences(args, batchesCoupling, embeddingTuple)
			inferInput = [batch[0] for batch in batches]
			inferInputLength = [batch[2] for batch in batches]
			correctOutput = [batch[1] for batch in batches] 
			inferInput = zip(inferInput, inferInputLength)
			args.print_verbose("Go with correctOutput and proper batch")
		else:
			inferInput, correctIndex = generateInferenceInputFromFile(args, embeddingTuple)
			correctOutput = None
			args.print_verbose("Go without correctOutput")
		# data will be returned as (input - inputLength) list of batches, (correctOutput) list of batches, and reordering indexes
		return list(inferInput), correctOutput, correctIndex
	else:
		raise ArgumentTypeError("args.mode unrecognized, must be train/infer: {:s}".format(args.mode))

if __name__ == "__main__":
	args = constructParser()
	if(args.load_params or args.save_params):
		tryLoadOrSaveParams(args, ['mode', 'directory'])
	if(args.mode == 'infer'):
		args.dropout = 1.0
	if(args.learning_rate is None):
		args.learning_rate = 0.001 if(args.optimizer == 'adam') else 1.0
	# args.directory = 'data\\vietchina'
	# args.src_dict_file = 'vi_tokenized.embedding.bin'
	# args.tgt_dict_file = 'ch_tokenized.embedding.bin'
	args.src_dict_file = args.src + args.embedding_file_name if(args.prefix) else args.embedding_file_name + '.' + args.src
	args.tgt_dict_file = args.tgt + args.embedding_file_name if(args.prefix) else args.embedding_file_name + '.' + args.tgt
	# args.src_file = 'vi_tokenized.txt'
	# args.tgt_file = 'ch_tokenized.txt'
	# args.size_hidden_layer = 128
	# args.initialize_range = (-1.0, 1.0)
	# args.save_path = 'save\\initEmbedding'
	# args.epoch = 100
	# args.evaluation_step = 20
	args.global_steps = 0
	# args.import_default_dict = True
	# args.maximum_sentence_length = 50
	# args.batch_size = 128
	
	if(args.verbose):
		def verbose(*argv, **kwargs):
			print(*argv, **kwargs)
	else:
		def verbose(*argv, **kwargs):
			pass
	args.print_verbose = verbose
	if(args.read_mode == 'vocab'):
		# in vocab mode, must train the embedding as well
		args.train_embedding = True
	## REMOVE. JUST USE REQUIRED
	if(args.save_path is None):
		raise argparse.ArgumentTypeError("No save path for anything. Exiting.")
	args.src_file = args.src + args.input_file_name if(args.prefix) else args.input_file_name + '.' + args.src
	args.tgt_file = args.tgt + args.input_file_name if(args.prefix) else args.input_file_name + '.' + args.tgt
	
	timer = time.time()
	def getTimer():
		return time.time()-timer
	args.time_passed = getTimer
	
	# Create the session here
	sessionTuple, embeddingTuple = constructAndLoadSession(args)
	# extract the values from the tuples
	session, inputOutputTuple, configTuple, trainTuple = sessionTuple
	srcEmbeddingTuple, tgtEmbeddingTuple = embeddingTuple
	srcWordToId, srcIdToWord, _ = srcEmbeddingTuple
	tgtWordToId, tgtIdToWord, _ = tgtEmbeddingTuple
	# testRun(args, sessionTuple, embeddingTuple)
	
	data = loadDataByMode(args, embeddingTuple)
	if(args.mode == 'train'):
		# Try to load created batch file as default
		batches, sample = data
		# create random idx to print a sample consistently during evaluation
		rIdx1, rIdx2 = np.random.randint(len(sample[2]), size=2)
		endTokenId = tgtWordToId[args.end_token]
		def evaluationFunction(extraArgs):
			iteration, losses, prevBest = extraArgs
			trainResult, inferResult = evaluateSession(args, sessionTuple, embeddingTuple, sample)
			_, correctOutput, _, trimLength, trainInput = sample
			print(stripResultArray(trainInput[rIdx1], endTokenId), '\n=> TRAIN: ', stripResultArray(trainResult[rIdx1], endTokenId), '\n= ', stripResultArray(correctOutput[rIdx1], endTokenId))
			print('=> INFER: ', stripResultArray(inferResult[rIdx1], endTokenId), '\n= ', stripResultArray(correctOutput[rIdx1], endTokenId))
			print(stripResultArray(trainInput[rIdx2], endTokenId), '\n=> TRAIN: ', stripResultArray(trainResult[rIdx2], endTokenId), '\n= ', stripResultArray(correctOutput[rIdx2], endTokenId))
			print('=> INFER: ', stripResultArray(inferResult[rIdx2], endTokenId), '\n= ', stripResultArray(correctOutput[rIdx2], endTokenId))
			trainResult = calculateBleu(correctOutput, trainResult, trimLength)
			inferResult = calculateBleu(correctOutput, inferResult, trimLength)
			print("Iteration %d, time passed %.2fs, BLEU score %2.2f(@train) and %2.2f(@infer) " % (iteration, getTimer(), trainResult * 100.0, inferResult * 100.0))
			print("Losses during this cycle: {}".format(losses[-args.evaluation_step:]))
			if(args.save_when_evaluate):
				if(prevBest < inferResult):
					bestPath = args.save_path + ".best"
					builder.saveToPath(session, bestPath)
					print("Save best eval model to {:s}.".format(bestPath))
					return inferResult
				else:
					return prevBest
			return -1.0
		# evaluate the initialized model. Expect horrendous result
		evaluationFunction((0, [], -1.0))
		# execute training
		totalLossTrack = trainSession(args, sessionTuple, batches, evaluationFunction)
	elif(args.mode == 'infer'):
		# infer will try to read input in file input.src and output to file output.tgt
		# INCOMPLETED
		inferInput, correctData, reorderIdx = data
		inferOutput = inferenceSession(args, sessionTuple, inferInput)
		args.print_verbose("Sample @idx=[0], first batch:", inferInput[0][0][0], '\n=>', inferOutput[0][0])
		
		# Flatten the inferOutput
		inferOutput = np.concatenate(inferOutput, axis=0)
		outputFile = outputInferenceToFile(args, embeddingTuple, inferOutput, reorderIdx=reorderIdx)
		
		if(correctOutput):
			# Flatten the correctOutput and trimLength as well
			# correctOutput = np.concatenate(correctOutput, axis=0)
			correctOutput = [item for sublist in correctOutput for item in sublist]
			trimLength = np.concatenate([batch[3] for batch in batches])
			# print(np.shape(correctOutput), np.shape(inferOutput), np.shape(trimLength))
			inferResult = calculateBleu(correctOutput, inferOutput, trimLength)
			print("Inference mode ran and saved to %s, BLEU score %2.2f, time passed %.2fs" % (outputFile, inferResult * 100.0, getTimer()))
		else:
			print("Inference mode ran and saved to %s, time passed %.2fs" % (outputFile, getTimer()))
	else:
		raise argparse.ArgumentTypeError("Invalid mode for main flow, must be train/infer, but is {:s}.".format(args.mode))
	saveSession(args, sessionTuple)
	print("All task completed, total time passed %.2fs" % getTimer())
