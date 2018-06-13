from ffBuilder import *
from exampleBuilder import *
from itertools import islice
from random import randint
import numpy as np
import tensorflow as tf
import io, re, argparse, os, time

trainPacRegex = re.compile("PAC \d+_\d+, \((.+) (.+) (.+)\) \((.+) (.+) (.+)\) ([\d-]+) ([\d-]+) (\d+)")
referencePacRegex = re.compile("PAC \d+_\d+, \((.+) (.+) (.+)\) \((.+) (.+) (.+)\) ([\d-]+) ([\d-]+)")
trainSibRegex = re.compile("SIB \d+_\d+, \((.+) (.+) (.+) ([\d-]+)\) \((.+) (.+) (.+) ([\d-]+)\) \((.+) (.+)\) ([\d-]+) (\d+)")
referenceSibRegex = re.compile("SIB \d+_\d+, \((.+) (.+) (.+) ([\d-]+)\) \((.+) (.+) (.+) ([\d-]+)\) \((.+) (.+)\) ([\d-]+)")

def createDualModel(wordVectorSize, loadPreviousModel=None, hidden_layers=[300], tagSizeTuple=(1,1)):
	posTagSize, dependencySize = tagSizeTuple
	pacModelSize = wordVectorSize + posTagSize + dependencySize + wordVectorSize + posTagSize + dependencySize + 2
	sibModelSize = wordVectorSize + posTagSize + dependencySize + 1 + wordVectorSize + posTagSize + dependencySize + 1 + wordVectorSize + posTagSize + 1
	
	# Destroy any current graph and session
	tf.reset_default_graph()
	session = tf.get_default_session()
	if(session is not None):
		sess.close()
	
	# Initialize graph connections
	session = tf.Session()
	session, pac_train_op, pac_prediction, pac_input, pac_output = createTensorflowSession(pacModelSize, 1, 'PAC_', 0.5, hidden_layers, session)
	session, sib_train_op, sib_prediction, sib_input, sib_output = createTensorflowSession(sibModelSize, 1, 'SIB_', 0.5, hidden_layers, session)
	session.run(tf.global_variables_initializer())
	
	if(isinstance(loadPreviousModel, str)):
		# Load the weight in saved file
		print("Load previous model from {}".format(loadPreviousModel))
		loadFromPath(session, loadPreviousModel)
		# print(tf.trainable_variables())
	else:
		# Create new model from scratch here instead
		# TODO later
		print("Create new model")
	
	# Return relating variables to the session
	return session, (pac_train_op, pac_prediction, pac_input, pac_output), (sib_train_op, sib_prediction, sib_input, sib_output)

def trainDualModel(sessionTuple, pacDataTuple, sibDataTuple, epoch=100):
	session, pac_tuple, sib_tuple = sessionTuple
	
	train_op, _, input, output = pac_tuple
	runTrainingForSession(session, (train_op, input, output), pacDataTuple, epoch)
	train_op, _, input, output = sib_tuple
	runTrainingForSession(session, (train_op, input, output), sibDataTuple, epoch)
	
def runDualModel(sessionTuple, pacDataTuple, sibDataTuple, labeling=True):
	session, pac_tuple, sib_tuple = sessionTuple
	_, pac_prediction, pac_input, _ = pac_tuple
	_, sib_prediction, sib_input, _ = sib_tuple
	
	if(len(pacDataTuple[0]) > 0):
		pac_result = runWorkingSession(session, (pac_prediction, pac_input), pacDataTuple[0])
		if(labeling):
			pac_result = [(pacDataTuple[1][i], pac_result[i]) for i in range(len(pac_result))]
		#	pac_result[i] = (pacDataTuple[1][i], pac_result[i])
	else:
		pac_result = []
	
	if(len(sibDataTuple[0]) > 0):
		sib_result = runWorkingSession(session, (sib_prediction, sib_input), sibDataTuple[0])
		if(labeling):
			sib_result = [(sibDataTuple[1][i], sib_result[i]) for i in range(len(sib_result))]
		#	sib_result[i] = (sibDataTuple[1][i], sib_result[i])
	else:
		sib_result = []
	
	return pac_result, sib_result
	
def convertBlockToVectors(block, tagCombinedDict, wordVectorDict, mode, tagMode='normal', defaultTuple=None, exclusionFilter=None):
	# Input an array of lines and return splitted PAC/SIB array containing respective vector 
	# In train mode, return tuple of data-result vector; in ref mode, return tuple of data-line
	if(defaultTuple is not None):
		defaultTagId, defaultDetId, defaultWordVector = defaultTuple
		defaultWordVector = wordVectorDict[defaultWordVector]
	if(mode == 'train'):
		pacRegex = trainPacRegex
		sibRegex = trainSibRegex
	else:
		pacRegex = referencePacRegex
		sibRegex = referenceSibRegex

	pacData = ([],[])
	sibData = ([],[])
	
	#counter = randint(1,len(block)) - 1
	# tagMode onehot or normal, adding tags as normal or as onehot
	def handleNormalTag(input, idx):
		input.append(idx)
	def handleOneHotTag(input, vector):
		input.extend(vector)
	if(tagMode == 'normal'):
		appendData = handleNormalTag
	else:
		appendData = handleOneHotTag
	
	for line in block:
		#counter -= 1
		# Get correct type for line
		if(line.find('PAC') ==0):
			match = re.match(pacRegex, line)
			if(match is not None):
				fullInput = []
				if(defaultTuple is None):
					# Error and discard if value not found
					try:
						fullInput.extend(wordVectorDict[match.group(1)])
						appendData(fullInput, tagCombinedDict[match.group(2)])
						appendData(fullInput, tagCombinedDict[match.group(3)])
						# subtitute w2
						fullInput.extend(wordVectorDict[match.group(4)])
						appendData(fullInput, tagCombinedDict[match.group(5)])
						appendData(fullInput, tagCombinedDict[match.group(6)])
						# distance and punctuation mark inbetween
						fullInput.append(float(match.group(7)))
						fullInput.append(float(match.group(8)))
						# Add into pacData
						pacData[0].append(fullInput)
						if(mode == 'train'):
							output = [float(match.group(9))]
							if(exclusionFilter):
								if(match.group(1) in exclusionFilter or match.group(5) in exclusionFilter):
									output.append(False)
								else:
									output.append(True)
							pacData[1].append(output)
						else:
							pacData[1].append(line)
					except KeyError:
						print("KeyError (no default value, @PAC), line: {}".format(line))
						print(match.group(1) in wordVectorDict, match.group(2) in tagCombinedDict, match.group(3) in tagCombinedDict,
						match.group(4) in wordVectorDict, match.group(5) in tagCombinedDict, match.group(6) in tagCombinedDict)
				else:
					try:
						# Have default value, subtitute if word not found
						# subtitute w1
						fullInput.extend(wordVectorDict.get(match.group(1), defaultWordVector))
						appendData(fullInput, tagCombinedDict.get(match.group(2), defaultTagId))
						appendData(fullInput, tagCombinedDict.get(match.group(3), defaultDetId))
						# subtitute w2
						fullInput.extend(wordVectorDict.get(match.group(4), defaultWordVector))
						appendData(fullInput, tagCombinedDict.get(match.group(5), defaultTagId))
						appendData(fullInput, tagCombinedDict.get(match.group(6), defaultDetId))
						# distance and punctuation mark inbetween
						fullInput.append(float(match.group(7)))
						fullInput.append(float(match.group(8)))
						# Add into pacData
						pacData[0].append(fullInput)
						if(mode == 'train'):
							output = [float(match.group(9))]
							if(exclusionFilter):
								if(match.group(1) in exclusionFilter or match.group(5) in exclusionFilter):
									output.append(False)
								else:
									output.append(True)
							pacData[1].append(output)
						else:
							pacData[1].append(line)
					except KeyError:
						print("KeyError (with default value, @PAC), line: {}".format(line))
						print(match.group(1) in wordVectorDict, match.group(2) in tagCombinedDict, match.group(3) in tagCombinedDict,
							match.group(4) in wordVectorDict, match.group(5) in tagCombinedDict, match.group(6) in tagCombinedDict)
		
		elif(line.find('SIB') ==0):
			match = re.match(sibRegex, line)
			if(match is not None):
				fullInput = []
				if(defaultTuple is None):
					# Error and discard if value not found
					try:
						fullInput.extend(wordVectorDict[match.group(1)])
						appendData(fullInput, tagCombinedDict[match.group(2)])
						appendData(fullInput, tagCombinedDict[match.group(3)])
						fullInput.append(float(match.group(4)))
						# subtitute w2
						fullInput.extend(wordVectorDict[match.group(5)])
						appendData(fullInput, tagCombinedDict[match.group(6)])
						appendData(fullInput, tagCombinedDict[match.group(7)])
						fullInput.append(float(match.group(8)))
						# subtitute parent
						fullInput.extend(wordVectorDict[match.group(9)])
						appendData(fullInput, tagCombinedDict[match.group(10)])
						# punctuation inbetween
						fullInput.append(float(match.group(11)))
						# Add into sibData
						sibData[0].append(fullInput)
						if(mode == 'train'):
							output = [float(match.group(12))]
							if(exclusionFilter):
								if(match.group(1) in exclusionFilter or match.group(5) in exclusionFilter):
									output.append(False)
								else:
									output.append(True)
							sibData[1].append(output)
						else:
							sibData[1].append(line)
					except KeyError:
						print("KeyError (no default value, @SIB), line: {}".format(line))
						print(match.group(1) in wordVectorDict, match.group(2) in tagCombinedDict, match.group(3) in tagCombinedDict,
							match.group(5) in wordVectorDict, match.group(6) in tagCombinedDict, match.group(7) in tagCombinedDict, match.group(9) in wordVectorDict, match.group(10) in tagCombinedDict)
				else:
					try:
						# Have default value, subtitute if word not found
						# subtitute w1
						fullInput.extend(wordVectorDict.get(match.group(1), defaultWordVector))
						appendData(fullInput, tagCombinedDict.get(match.group(2), defaultTagId))
						appendData(fullInput, tagCombinedDict.get(match.group(3), defaultDetId))
						fullInput.append(float(match.group(4)))
						# subtitute w2
						fullInput.extend(wordVectorDict.get(match.group(5), defaultWordVector))
						appendData(fullInput, tagCombinedDict.get(match.group(6), defaultTagId))
						appendData(fullInput, tagCombinedDict.get(match.group(7), defaultDetId))
						fullInput.append(float(match.group(8)))
						# subtitute parent
						fullInput.extend(wordVectorDict.get(match.group(9), defaultWordVector))
						appendData(fullInput, tagCombinedDict.get(match.group(10), defaultTagId))
						# punctuation inbetween
						fullInput.append(float(match.group(11)))
						# Add into sibData
						sibData[0].append(fullInput)
						if(mode == 'train'):
							output = [float(match.group(12))]
							if(exclusionFilter):
								if(match.group(1) in exclusionFilter or match.group(5) in exclusionFilter):
									output.append(False)
								else:
									output.append(True)
							sibData[1].append(output)
						else:
							sibData[1].append(line)
					except KeyError:
						print("KeyError (with default value, @SIB), line: {}".format(line))
						print(match.group(1) in wordVectorDict, match.group(2) in tagCombinedDict, match.group(3) in tagCombinedDict,
							match.group(5) in wordVectorDict, match.group(6) in tagCombinedDict, match.group(7) in tagCombinedDict, match.group(9) in wordVectorDict, match.group(10) in tagCombinedDict)
		#if(counter == 0):
			# Randomly output a value set for each block
			#print("Line: {}\nMatch: {}\nInputData:{}".format(line, match, fullInput))
	return pacData, sibData

def nodeRelationToVector(relation, nodeTuple, dictTuple, defaultTuple=None, excludedWords=None):
	# Require tagDict to return array values
	combinedTagDict, wordVectorDict = dictTuple
	if(defaultTuple is not None):
		defaultTagId, defaultDetId, defaultWordVector = defaultTuple
		defaultWordVector = wordVectorDict[defaultWordVector]
	
	if(relation == 'PAC'):
		parent, child = nodeTuple
		if(defaultTuple is not None):
			result = list(wordVectorDict.get(parent.word, defaultWordVector))
			result.extend(combinedTagDict.get(parent.tag, defaultTagId))
			result.extend(combinedTagDict.get(parent.dependency, defaultDetId))
			result.extend(wordVectorDict.get(child.word, defaultWordVector))
			result.extend(combinedTagDict.get(child.tag, defaultTagId))
			result.extend(combinedTagDict.get(child.dependency, defaultDetId))
			result.append(float(getPunctuationInBetween(child, parent)))
			result.append(float(getDistance(child, parent)))
		else:
			try:
				result = list(wordVectorDict[parent.word])
				result.extend(combinedTagDict[parent.tag])
				result.extend(combinedTagDict[parent.dependency])
				result.extend(wordVectorDict[child.word])
				result.extend(combinedTagDict[child.tag])
				result.extend(combinedTagDict[child.dependency])
				result.append(float(getPunctuationInBetween(child, parent)))
				result.append(float(getDistance(child, parent)))
			except KeyError:
				print("PAC key error, data {} {} {} {} {} {}".format(parent.word, parent.tag, parent.dependency, child.word, child.tag, child.dependency))
				print(parent.word in wordVectorDict, parent.tag in combinedTagDict, parent.dependency in combinedTagDict,
					child.word in wordVectorDict, child.tag in combinedTagDict, child.dependency in combinedTagDict)
		return result
	elif(relation == 'SIB'):
		one, other, parent = nodeTuple
		if(defaultTuple is not None):
			result = list(wordVectorDict.get(one.word, defaultWordVector))
			result.extend(combinedTagDict.get(one.tag, defaultTagId))
			result.extend(combinedTagDict.get(one.dependency, defaultDetId))
			result.append(float(getDistance(one, parent)))
			result.extend(wordVectorDict.get(other.word, defaultWordVector))
			result.extend(combinedTagDict.get(other.tag, defaultTagId))
			result.extend(combinedTagDict.get(other.dependency, defaultDetId))
			result.append(float(getDistance(other, parent)))
			result.extend(wordVectorDict.get(parent.word, defaultWordVector))
			result.extend(combinedTagDict.get(parent.tag, defaultTagId))
			result.append(float(getPunctuationInBetween(one, parent, other)))
		else:
			try:
				result = list(wordVectorDict[one.word])
				result.extend(combinedTagDict[one.tag])
				result.extend(combinedTagDict[one.dependency])
				result.append(float(getDistance(one, parent)))
				result.extend(wordVectorDict[other.word])
				result.extend(combinedTagDict[other.tag])
				result.extend(combinedTagDict[other.dependency])
				result.append(float(getDistance(other, parent)))
				result.extend(wordVectorDict[parent.word])
				result.extend(combinedTagDict[parent.tag])
				result.append(float(getPunctuationInBetween(one, parent, other)))
			except KeyError:
				print("SIB key error, data {} {} {} {} {} {} {} {}".format(parent.word, parent.tag, parent.dependency, child.word, child.tag, child.dependency))
				print(one.word in wordVectorDict, one.tag in combinedTagDict, one.dependency in combinedTagDict,
					other.word in wordVectorDict, other.tag in combinedTagDict, other.dependency in combinedTagDict, parent.word in wordVectorDict, parent.tag in tagCombinedDict)
		return result

def generateMirrorCases(trainingCases, lengthTuple, isTraining=True):
	# Currently ignore the cases with punctuation inbetween and cases marked with False in result
	# isTraining variable if false will output function with referencing trainingCases
	pacData, sibData = trainingCases
	wordVectorSize, posTagSize, dependencySize = lengthTuple
	
	# Handle pacData tuple
	pacTrainCases, pacResult = pacData
	newTrainingCases = []
	newResult = []
	# PAC punctuation position is -2 (before last)
	for i in range(len(pacTrainCases)):
		if(pacTrainCases[i][-2] == 0 or (len(pacResult[i]) == 2 and pacResult[i][1])):
			# Acceptable cases to switch
			newCase = list(pacTrainCases[i])
			# Negate the value of 'signed distance' at -1 (last)
			newCase[-1] = -newCase[-1]
			# Add into newTrainingCases
			newTrainingCases.append(newCase)
			if(isTraining):
				newResult.append([ 1.0 - pacResult[i][0] ])
			else:
				newResult.append(i)
	# Append data back into package
	if(isTraining):
		pacTrainCases.extend(newTrainingCases)
		pacResult.extend(newResult)
	else:
		pacNewCases = (newTrainingCases, newResult)
	totalMirorCases = len(newResult)
	# print("PAC mirror case generated {}".format(len(newResult)))
	
	# Handle sibData tuple
	sibTrainCases, sibResult = sibData
	newTrainingCases = []
	newResult = []
	wordSize = wordVectorSize + posTagSize + dependencySize + 1
	# SIB punctuation position is -1 (last)
	for i in range(len(sibTrainCases)):
		if(sibTrainCases[i][-1] == 0 or (len(sibResult[i]) == 2 and sibResult[i][1])):
			# Acceptable cases to switch
			# Swap the left and right word, leaving distance to parent aside
			# First, take rightword as left word, copy distance as-is
			newCase = sibTrainCases[i][wordSize:(wordSize*2-1)]
			newCase.append(sibTrainCases[i][wordSize-1])
			# Take left word as right word
			newCase.extend(sibTrainCases[i][:(wordSize-1)])
			# Append the rest to newCase
			newCase.extend(sibTrainCases[i][len(newCase):len(sibTrainCases[i])])
			# Add into newTrainingCases
			newTrainingCases.append(newCase)
			if(isTraining):
				newResult.append([ 1.0 - sibResult[i][0] ])
			else:
				newResult.append(i)
	# Append data back into package
	if(isTraining):
		sibTrainCases.extend(newTrainingCases)
		sibResult.extend(newResult)
	else:
		sibNewCases = (newTrainingCases, newResult)
	# print("SIB mirror case generated {}".format(len(newResult)))
	totalMirorCases += len(newResult)
	if(not isTraining):
		return (pacNewCases, sibNewCases)
	else:
		return totalMirorCases

def rearrangeTree(treeRoot, switchCases):
	pacData, sibData = switchCases
	# Compress the array
	pacSwitchCases = [(pacCase[0][0], pacCase[0][1], pacCase[1][0]) for pacCase in pacData]
	sibSwitchCases = [(sibCase[0][0], sibCase[0][1], sibCase[1][0]) for sibCase in sibData]
	#for parent, child, val in sibSwitchCases:
	#	print("{}({})-{}({}): {}".format(parent.word, parent.pos, child.word, child.pos, val))
	# print("------")
	def labelChildren(node):
		if(node.isPunctuation()):
			return
		# An exception - treeRoot will not have any PAC relation and will be leftmost
		# Seperate the children into left and right side
		if(node is not treeRoot):
			leftSide = []
			rightSide = []
			for child in node.children:
				# Search for the switch, should exist 
				switch = next((swi for par, chi, swi in pacSwitchCases if par is node and chi is child), None)
				# If not found, likely to be an error, print out
				if(switch is None):
					switch = 0.0
					if(not child.isPunctuation()):
						print("Not found relation for PAC: parent {} {}, child {} {}".format(node.word, node.pos, child.word, child.pos))
				if((switch < 0.5 and child.pos < node.pos) or (switch > 0.5 and child.pos > node.pos)):
					leftSide.append(child)
				else:
					rightSide.append(child)
		else:
			leftSide = []
			rightSide = list(node.children)
		# print('{} {} {}'.format(leftSide, node.word, rightSide))
		# For each side, write relative position to the parent node to the children
		while(len(leftSide) > 0):
			leftmost = leftSide[0]
			for sib in leftSide:
				# Compare between sibling node
				if(sib is leftmost):
					continue
				currentLeft, currentRight = (leftmost, sib) if(leftmost.pos < sib.pos) else (sib, leftmost)
				switch = next((swi for one, other, swi in sibSwitchCases if one is currentLeft and other is currentRight), None)
				if(switch is None):
					switch = 0.0
					if(not leftmost.isPunctuation() and not sib.isPunctuation()):
						print("Not found relation for SIB: current_left {} {}, sibling {} {}".format(leftmost.word, leftmost.pos, sib.word, sib.pos))
				if((switch < 0.5 and sib.pos < leftmost.pos) or (switch > 0.5 and sib.pos > leftmost.pos)):
					leftmost = sib
			# Write the relative position into otherTag
			leftmost.otherTag = - len(leftSide)
			# print("Left: [{}] {} to [{}]".format(leftmost.word, leftmost.otherTag, leftmost.parent.word))
			# Pop the processed node
			leftSide.remove(leftmost)
		# Right side
		while(len(rightSide) > 0):
			rightmost = rightSide[0]
			for sib in rightSide:
				# Compare between sibling node
				if(sib is rightmost):
					continue
				currentLeft, currentRight = (rightmost, sib) if(rightmost.pos < sib.pos) else (sib, rightmost)
				switch = next((swi for one, other, swi in sibSwitchCases if one is currentLeft and other is currentRight), None)
				if(switch is None):
					switch = 0.0
					if(not rightmost.isPunctuation() and not sib.isPunctuation()):
						print("Not found relation for SIB: current_right {} {}, sibling {} {}".format(rightmost.word, rightmost.pos, sib.word, sib.pos))
				if((switch < 0.5 and sib.pos > rightmost.pos) or (switch > 0.5 and sib.pos < rightmost.pos)):
					rightmost = sib
			# Write the relative position into otherTag
			rightmost.otherTag = len(rightSide)
			# print("Right: [{}] {} to [{}]".format(rightmost.word, rightmost.otherTag, rightmost.parent.word))
			# Pop the processed node
			rightSide.remove(rightmost)
	recursiveRun(labelChildren, treeRoot)
	# With the relative positions, reindex the tree into form of an array
	result = []
	treeRoot.otherTag = 0
	# print(treeRoot.children)
	def indexing(node, nextFunc):
		if(len(node.children) == 0):
			result.append(node)
			return
		full = [node]
		full.extend(node.children)
		try:
			full = sorted(full, key=lambda n: n.otherTag if(n is not node and isinstance(n.otherTag, int)) else 0 if(n is node) else -1 if(n.pos < node.pos) else 1)
		except TypeError:
			print("TypeError @indexing, array : ")
			for n in full:
				print("[{} {}]".format(n.word, n.otherTag))
				sys.exit()
		'''print('[')
		for unit in full:
			print("{}({} {}) ".format(unit.word, unit.pos, unit.otherTag))
		print(']')'''
		for unit in full:
			if(unit is not node):
				nextFunc(unit, nextFunc)
			else:
				result.append(node)
	indexing(treeRoot, indexing)
	return result

def getVectorDictFromFile(fileDir, refDict, generateMode='normal', defaultKey=None):
	if(defaultKey is not None):
		refDict[defaultKey] = 1
	refDict.pop(None, None)
	if(generateMode=='combine'):
		# Generate all keys needed
		altKey = {}
		for key in refDict:
			if(key.find('_') >= 0):
				# combined word, check for subword
				for newKey in '_'.split(key):
					newKey = newKey.lower()
					altKey[newKey] = altKey.get(newKey, 0) + 1
			altKey[key] = altKey.get(key, 0) + refDict[key]
	else:
		altKey = refDict
	# Generate minimal dictionary
	vectorDict = createMinimalWordDictFromFile(fileDir, altKey)
	# Convert back into correct vectorDict 
	if(generateMode=='combine'):
		correctVectorDict = {}
		for key in refDict:
			if(key in vectorDict):
				correctVectorDict[key] = vectorDict[key]
				continue
			correctVector = None
			if(key.find('_') >= 0):
				# combined word, check for subword
				for newKey in '_'.split(key):
					newKey = newKey.lower()
					if(newKey in vectorDict):
						# Found subword
						if(correctVector is None):
							# Copy directly if first
							correctVector = list(vectorDict[newKey])
						else:
							# Add into vector if other
							for i in range(len(correctVector)):
								correctVector[i] += vectorDict[newKey][i]
					else:
						# Not found subword, revert to None and escape
						correctVector = None
						break
				# if all subword found, add vector into dict, if not, discard
				if(correctVector is not None):
					correctVectorDict[key] = correctVector
				#else:
					# Discard, print warning
					# print("Key {} not found in vectorDict".format(key))
			else:
				# Word not found, ignore
				# print("Key {} not found in vectorDict, nor splitable".format(key))
				continue
		return correctVectorDict
	else:
		return vectorDict

def getRefDictFromFile(fileDir):
	file = io.open(fileDir, 'r', encoding='utf-8')
	refDict = {}
	for line in file.readlines():
		if(line.find('PAC') == 0):
			match = re.match(referencePacRegex, line)
			if(match is None):
				print("Error parsing RefDict, line {}".format(line))
			keys = [match.group(1), match.group(4)]
		elif(line.find('SIB') == 0):
			match = re.match(referenceSibRegex, line)
			if(match is None):
				print("Error parsing RefDict, line {}".format(line))
				continue
			keys = [match.group(1), match.group(5), match.group(9)]
		else:
			keys = None
		if(keys is not None):
			# Check found word
			for key in keys:
				refDict[key] = refDict.get(key, 0) + 1
	file.close()
	return refDict

def writeDataIntoFile(allDataTuple, fileDir):
	file = io.open(fileDir, 'w', encoding='utf-8')
	for line, result in allDataTuple:
		file.write("%s (%2.5f)\n" % (line.strip(), result))
	file.close()

def writeNodeArrayToFile(array, openedFile, format='%s '):
	for node in array:
		openedFile.write(format % node.word)

def printCurrentWeightToFile(session, fileDir):
	data = session.run(fetches=tf.trainable_variables())
	file = io.open(fileDir, 'w', encoding='utf-8')
	for weights in data:
		for weight in weights:
			weight.tofile(file, ' ', '%2.5f')
			#np.savetxt(file, weight)
			file.write('\n')
		file.write('\n# New slice\n')
	file.close()

def createOneHotVector(idx, size):
	vector = np.zeros(size)
	if(idx >= 0 and idx < size):
		vector[idx] = 1.0
	return vector.tolist()
	
def strAsArray(str):
	listHidden = str.split(',')
	try:
		listHidden = [int(layer) for layer in listHidden]
	except ValueError:
		print("Error during parsing, hidden layer default [300]")
		return [300]
	return listHidden
	
if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Build dual model to handle PAC/SIB.')
	#parser.add_argument('integers', metavar='N', type=int, nargs='+',
	#					help='an integer for the accumulator')
	#parser.add_argument('--src', type=str, default='ja', help='source extension')
	#parser.add_argument('--tgt', type=str, default='en', help='target extension')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input files')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-t','--tagdir', type=str, default="all_tag.txt", help='location of the tag file containing both POStag and dependency, default all_tag.txt')
	parser.add_argument('-e','--embedding', type=str, required=True, help='embedding word file')
	parser.add_argument('--embedding_mode', type=str, default="normal", help='type of word used in training file vs embedding_word file, normal|combine, default normal')
	parser.add_argument('--embedding_size', type=int, default=None, help='size of the word vector in the embedding file')
	parser.add_argument('--tag_mode', type=str, default="normal", help='mode to handle tag file, onehot|normal, default normal')
	parser.add_argument('-m','--mode', type=str, default="train", help='Running mode of the model, train|ref|reorder, default train')
	parser.add_argument('-s','--savedir', type=str, default=None, help='location of previous saved model')
	parser.add_argument('--default_word', type=str, default=None, help='key in embedding_word standing for unknown word')
	parser.add_argument('--epoch', type=int, default=100, help='number of training for each example')
	parser.add_argument('--batch_size', type=int, default=512, help='number of training per batch')
	parser.add_argument('--input_extension', type=str, default='in', help='extension for input file')
	parser.add_argument('--output_extension', type=str, default='out', help='extension for output file')
	parser.add_argument('--hidden_layer', type=strAsArray, default=[300], help='hidden layers construction, default 300')
	parser.add_argument('--log', action='store_true', help='put all weights out a .log file for comparison')
	parser.add_argument('--mirror', action='store_true', help='generate mirror cases for training and reorder phases')
	parser.add_argument('--mirror_exclusion', type=str, default=None, help='file detailing all words that will be excluded during mirroring')
	parser.add_argument('--output_word_str', type=str, default="[%s]", help='format of individual words in reference mode, default [%%s]. Subtitute space as _ in the command line')
	parser.add_argument('--no_old_sentence', action='store_false', help='do not print old sentences on the out file')
	parser.add_argument('--no_root', action='store_false', help='do not print the root node in the out file')
	args = parser.parse_args()
	args.split = 0.5
	# args.embedding_word = "data\syntacticEmbeddings\skipdep_embeddings.txt"
	# args.default_word = "*UNKNOWN*"
	args.output_word_str = args.output_word_str.replace('_', ' ')
	if(args.outputdir is None):
		args.outputdir = args.inputdir
	if(args.embedding_size is None):
		# Read the vector size from the first line of the file
		file = io.open(args.embedding, 'r', encoding='utf-8')
		firstLine = file.readlines()[0]
		args.embedding_size = int(firstLine.split( )[1])
		file.close()
		print("Read embedding vector size: {}".format(args.embedding_size))
	if(args.mirror_exclusion):
		# Read the file into an array and load it into argument
		file = io.open(args.mirror_exclusion, 'r', encoding='utf-8')
		items = file.readlines()
		items = [item.strip() for item in items if(not re.match(blankLineRegex, item))]
		file.close()
		args.mirror_exclusion = items
	
	timer = time.time()
	# Change relating variables
	if(args.log):
		logDir = args.outputdir + '.log'
	args.inputdir = args.inputdir + '.' + args.input_extension
	args.outputdir = args.outputdir + '.' + args.output_extension
	# First, load tagDict
	allTagDict = getTagFromFile(args.tagdir)
	combinedTagDict = {}
	if(args.tag_mode == 'onehot'):
		posTagSize = len(allTagDict[0])
		dependencySize = len(allTagDict[1])
		for key in allTagDict[0]:
			combinedTagDict[key] = createOneHotVector(allTagDict[0][key], posTagSize)
		for key in allTagDict[1]:
			combinedTagDict[key] = createOneHotVector(allTagDict[1][key], dependencySize)
		print("One hot size: {} {}".format(posTagSize, dependencySize))
	else:
		posTagSize = 1
		dependencySize = 1
		combinedTagDict.update(allTagDict[0])
		combinedTagDict.update(allTagDict[1])
	# defaultTuple is decided depending on args.tag_mode as well
	if(args.default_word is None):
		defaultTuple = None
	else:
		if(args.tag_mode == 'onehot'):
			defaultTuple = (createOneHotVector(-1, posTagSize), createOneHotVector(-1, dependencySize), args.default_word)
		else:
			defaultTuple = (-1, -1, args.default_word)
	# Load previous model or create new
	sessionTuple = createDualModel(args.embedding_size, args.savedir, args.hidden_layer, (posTagSize, dependencySize))
	session, pac_tuple, sib_tuple = sessionTuple
	# Prepare values to convert data
	refDict = getRefDictFromFile(args.inputdir)
	refDict[args.default_word] = 1
	vectorDict = getVectorDictFromFile(args.embedding, refDict, args.embedding_mode)
	print("Done read tagDict and vectorDict, time passed %.2fs" % (time.time() - timer))
	
	# Begin dealing with data
	inputFile = io.open(args.inputdir, 'r', encoding='utf-8')
	if(args.mode == 'train'):
		counter = 0
		counterMilestone = 100
		caseCounter = 0
		if(args.mirror):
			mirrorCount = 0
		while True:
			counter += 1
			# Split files into blocks(batches) to be handled
			block = list(islice(inputFile, args.batch_size))
			if(not block):
				# Empty read, file already ended
				break
			# convert word data into vector data
			pacData, sibData = convertBlockToVectors(block, combinedTagDict, vectorDict, args.mode, args.tag_mode, defaultTuple, args.mirror_exclusion)
			# Generate mirror cases for certain circumstances
			if(args.mirror):
				mirrorCount += generateMirrorCases((pacData, sibData), (args.embedding_size, posTagSize, dependencySize))
			# filter the data back to acceptable output
			pacData = (pacData[0], [[x[0]] for x in pacData[1]])
			sibData = (sibData[0], [[x[0]] for x in sibData[1]])
			# Feed it in the training module
			trainDualModel(sessionTuple, pacData, sibData, args.epoch)
			caseCounter += len(pacData[1]) + len(sibData[1])
			if(counter % counterMilestone == 0):
				timePassed = time.time() - timer
				if(timePassed > 3600):
					print("Training for block %d(%d), total cases %d, time passed %dh%2dm" % (counter, len(block), caseCounter, int(timePassed) / 3600, (int(timePassed) / 60) % 60))
				elif(timePassed > 60):
					print("Training for block %d(%d), total cases %d, time passed %2dm%2.2fs" % (counter, len(block), caseCounter, int(timePassed) / 60, timePassed % 60))
				else:
					print("Training for block %d(%d), total cases %d, time passed %2.2fs" % (counter, len(block), caseCounter, time.time() - timer))
				if(args.mirror):
					print("Cases mirrored during last batch: {}".format(mirrorCount))
					mirrorCount = 0
		print("Done training, time passed %.2fs" % (time.time() - timer))
		# Save into a ckpt file
		if(args.savedir is None):
			# savedir not specified, save into local directory
			args.savedir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'ff.ckpt')
			print("Savedir not specified, save to default: {}".format(args.savedir))
		saveToPath(session, args.savedir)
		print("File saved, time passed %.2fs" % (time.time() - timer))
	elif(args.mode == 'ref'):
		# Convert into concerning dataset
		fileout = io.open(args.outputdir, 'w', encoding='utf-8')
		while True:
			block = list(islice(inputFile, args.batch_size))
			if(not block):
				break
			pacData, sibData = convertBlockToVectors(block, combinedTagDict, vectorDict, args.mode, args.tag_mode, defaultTuple)
			# run module
			pacResult, sibResult = runDualModel(sessionTuple, pacData, sibData)
			allResult = list(pacResult)
			allResult.extend(sibResult)
			# Write to output file
			for line, result in allResult:
				fileout.write("%s (%2.5f)\n" % (line.strip(), result))
		fileout.close()
		#writeDataIntoFile(allResult, args.outputdir)
		print("Reference ran, time passed %.2fs" % (time.time() - timer))
	elif(args.mode == 'test'):
		# pacData, sibData = convertBlockToVectors(inputFile.readlines(), combinedTagDict, vectorDict, 'train', args.tag_mode, defaultTuple)
		# run module
		# result = runTestTrainingSession(session, pac_tuple, pacData, 'PAC_prediction_error:0')
		# result.extend(runTestTrainingSession(session, sib_tuple, sibData, 'SIB_prediction_error:0'))
		# print(result)
		pacOneScore = 0.0
		pacZeroScore = 0.0
		pacOneCounter = 0
		pacZeroCounter = 0
		sibOneScore = 0.0
		sibZeroScore = 0.0
		sibOneCounter = 0
		sibZeroCounter = 0
		lineCounter = 0
		while True:
			block = list(islice(inputFile, args.batch_size))
			if(not block):
				break
			pacData, sibData = convertBlockToVectors(block, combinedTagDict, vectorDict, 'train', args.tag_mode, defaultTuple)
			pacResult, sibResult = runDualModel(sessionTuple, pacData, sibData)
			#if(lineCounter == 0):
			#	print(pacResult)
			#	print(sibResult)
			for i in range(len(pacResult)):
				# Compare result: if their difference is smaller than 0.5, add one (correct), else let score remain (incorrect)
				if(pacResult[i][0][0] == 0.0):
					pacZeroCounter += 1
					if(abs(pacResult[i][0][0] - pacResult[i][1][0]) < 0.5):
						pacZeroScore += 1.0
				else:
					pacOneCounter += 1
					if(abs(pacResult[i][0][0] - pacResult[i][1][0]) < 0.5):
						pacOneScore += 1.0
			for i in range(len(sibResult)):
				# Compare result: if their difference is smaller than 0.5, add one (correct), else let score remain (incorrect)
				if(sibResult[i][0][0] == 0.0):
					sibZeroCounter += 1
					if(abs(sibResult[i][0][0] - sibResult[i][1][0]) < 0.5):
						sibZeroScore += 1.0
				else:
					sibOneCounter += 1
					if(abs(sibResult[i][0][0] - sibResult[i][1][0]) < 0.5):
						sibOneScore += 1.0
			lineCounter += len(pacResult) + len(sibResult)
		# print("Score: PAC %2.2f%%, SIB %2.2f%%" % (pacFullScore / float(pacCounter) * 100, sibFullScore / float(sibCounter) * 100))
		print("Score: PAC %2.2f%%. 0: %2.2f%%(%d), 1: %2.2f%%(%d)\n       SIB %2.2f%%. 0: %2.2f%%(%d), 1: %2.2f%%(%d)" % 
		   ((pacOneScore + pacZeroScore) / float(pacOneCounter + pacZeroCounter) * 100, pacZeroScore / float(pacZeroCounter) * 100, pacZeroCounter, pacOneScore / float(pacOneCounter) * 100, pacOneCounter
		   , (sibOneScore + sibZeroScore) / float(sibOneCounter + sibZeroCounter) * 100, sibZeroScore / float(sibZeroCounter) * 100, sibZeroCounter, sibOneScore / float(sibOneCounter) * 100, sibOneCounter))
		print("Counted lines {}".format(lineCounter))
	elif(args.mode == 'reorder'):
		# Create trees from exampleBuilder function
		treeList = runConllParser(args.inputdir, None, None)
		# Define the recursive function going to be used and the array needed to store the cases
		pacData = ([], [])
		sibData = ([], [])
		if(args.tag_mode != 'onehot'):
			# Change defaultTuple and combinedTagDict into array
			defaultTuple = ([defaultTuple[0]],[defaultTuple[1]],defaultTuple[2])
			for key in combinedTagDict:
				combinedTagDict[key] = [combinedTagDict[key]]
		def getAllConnection(node, siblings):
			if(node.isPunctuation()):
				return
			# generate sib cases
			for other in siblings:
				if(other is node or other.isPunctuation() or other.pos < node.pos):
					continue
				sibData[0].append(nodeRelationToVector('SIB', (node, other, node.parent), (combinedTagDict, vectorDict), defaultTuple))
				sibData[1].append((node, other))
			# generate pac cases
			for child in node.children:
				if(child.isPunctuation()):
					continue
				pacData[0].append(nodeRelationToVector('PAC', (node, child), (combinedTagDict, vectorDict), defaultTuple))
				pacData[1].append((node, child))
		def getSiblings(node, child):
			return node.children
		# Run cases for each sentence
		fileout = io.open(args.outputdir, 'w', encoding='utf-8')
		firstLine = True
		for tree in treeList:
			# clear out pacData and sibData set
			del pacData[0][:]
			del pacData[1][:]
			del sibData[0][:]
			del sibData[1][:]
			# get all connections loaded into the arrays
			recursiveRun(getAllConnection, tree, [], getSiblings)
			# Generate result from existing model
			pacResult, sibResult = runDualModel(sessionTuple, pacData, sibData)
			# in mirror mode, cross-reference the opposite version for best decision
			if(args.mirror):
				# Run mirror cases
				# Cannot label the mirror exclusion here, manually ignore later
				pacMirror, sibMirror = generateMirrorCases((pacData, sibData), (args.embedding_size, posTagSize, dependencySize), False)
				pacMirror, sibMirror = runDualModel(sessionTuple, pacMirror, sibMirror)
				# try to cross-reference - compare idx for best result
				# PAC cases
				for i in range(len(pacMirror)):
					mirrorValue = float(pacMirror[i][1][0])
					caseIdx = pacMirror[i][0]
					if(args.mirror_exclusion is not None):
						# ignore cases with words in excluded list
						par, chi = pacResult[caseIdx][0]
						if(par.word in args.mirror_exclusion or chi.word in args.mirror_exclusion):
							continue
					pacResult[caseIdx] = (pacResult[caseIdx][0], [0.0 if(pacResult[caseIdx][1][0] < mirrorValue) else 1.0])
				# SIB cases
				for i in range(len(sibMirror)):
					mirrorValue = float(sibMirror[i][1][0])
					caseIdx = sibMirror[i][0]
					if(args.mirror_exclusion is not None):
						# ignore cases with words in excluded list
						one, other = sibResult[caseIdx][0]
						if(one.word in args.mirror_exclusion or other.word in args.mirror_exclusion):
							continue
					sibResult[caseIdx] = (sibResult[caseIdx][0], [0.0 if(sibResult[caseIdx][1][0] < mirrorValue) else 1.0])
			# rearrange tree based on the result, remove root from it if specified
			result = rearrangeTree(tree, (pacResult, sibResult))
			oldTree = sorted(tree.getAllNodeInTree(), key=lambda node: node.pos)
			if(args.no_root):
				result = filter(lambda node: node.pos != 0, result)
				oldTree = filter(lambda node: node.pos != 0, oldTree)
			# Write old and new version for comparison
			# End line character
			if(not firstLine):
				fileout.write("\n")
			else:
				firstLine = False
			# Write the old version if needed
			if(args.no_old_sentence):
				fileout.write("Old: ")
				writeNodeArrayToFile(oldTree, fileout, args.output_word_str)
				fileout.write("\nNew: ")
			# Write the new version
			writeNodeArrayToFile(result, fileout, args.output_word_str)
		fileout.close()
		print("Reorder ran, time passed %.2fs" % (time.time() - timer))
	else:
		raise argparse.ArgumentTypeError("Incorect mode, must be train|ref")
	if(args.log):
		printCurrentWeightToFile(session, logDir)
	session.close()