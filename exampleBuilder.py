import sys, re, argparse, io, time

blankLineRegex = re.compile("^\s*$")
PACFormatString = u"PAC {}, ({} {} {}) ({} {} {}) {} {} {}\n"
SIBFormatString = u"SIB {}, ({} {} {} {}) ({} {} {} {}) ({} {}) {} {}\n"

def getDataBlock(lines, checkRegex, keepRegexLine=False):
	allBlocks = []
	currentBlock = []
	for line in lines:
		if(checkRegex.search(line)):
			allBlocks.append(currentBlock)
			currentBlock = []
			if(keepRegexLine):
				currentBlock.append(line)
		else:
			currentBlock.append(line)
			
	return allBlocks
	
def filterBlock(allBlocks, sentenceRegex = None):
	# Used for stanford parser results
	newAllBlocks = []
	for listLines in allBlocks:
		save = False
		newBlock = []
		newOther = []
		if(sentenceRegex is not None):
			sentence = re.match(sentenceRegex, listLines[0]).group(1)
		for line in listLines:
			if(re.match(blankLineRegex, line)):
				save = True
			elif(save):
				newBlock.append(line)
			else:
				newOther.append(line)
		if(sentenceRegex is None):
			newAllBlocks.append((newBlock, newOther))
		else:
			newAllBlocks.append((newBlock, newOther, sentence))
	return newAllBlocks
	
def constructTreeFromBlock(lines, regexOrRegexTuple, anotateBlock=None, sentence=None):
	stanfordVersion = anotateBlock is not None
	if(stanfordVersion):
		regex, anotateRegex = regexOrRegexTuple
	else:
		regex = regexOrRegexTuple
	allConnection = [re.match(regex, line) for line in lines]
	allConnection = list(filter(lambda c: c is not None, allConnection))
	
	if(stanfordVersion):
		# Stanford version
		# Regex will get group dependency 1, parent (2-3) and child (4-5)
		allNode = [WordTree(match.group(4), [], int(match.group(5)), match.group(1)) for match in allConnection]
	else:
		# Conll version
		# Regex match as written: pos - word - POStag(full) - POStag - fatherPos - dependency
		# Will need a root Node to be created
		allNode = [WordTree(match.group(2), [], int(match.group(1)), match.group(6)) for match in allConnection]
		root = WordTree('root', [], 0, 'root')
		root.addPosTag('ROOT', 'ROOT')
		allNode.insert(0, root)
	
	root = next((x for x in allNode if x.pos == 0 and x.word == 'root'), None)
	for match in allConnection:
		try:
			if(stanfordVersion):
				parentIdx = int(match.group(4))
				childIdx = int(match.group(5))
			else:
				parentIdx = int(match.group(5))
				childIdx = int(match.group(1))
		except ValueError:
			print("Error during constructTreeFromBlock(ValueError), set {}".format(match))
		parent = next((x for x in allNode if x.pos == parentIdx), None)
		child = next((x for x in allNode if x.pos == childIdx), None)
		if(not stanfordVersion):
			child.addPosTag(match.group(4), match.group(3))
		if(parent is None):
			# Root found, take child as root (for parser)
			root = child
			if(parentIdx > 0 and False):
				print("Wrong root found, pos {} {}, root word {}".format(parentIdx, childIdx, root.word))
				print("ListConnections: {}".format(len(allConnection)))
				for con in allConnection:
					print(con.string)
		else:
			parent.children.append(child)
			child.addParent(parent)
	
	if(stanfordVersion):
		anotateTreeWithBlock(anotateBlock, anotateRegex, allNode)
		# use sentence to find all punctuation, then add them into root node as children
		counter = 0
		sentence = re.split(' ', sentence.strip())
		for word in sentence:
			counter += 1
			if(next((x for x in allNode if x.pos == counter), None) is None):
				# Node not found, try adding
				punct = WordTree(word, [], counter, WordTree.punctuation_dep)
				punct.addParent(root)
				root.children.append(punct)
				#if(word not in WordTree.listPunctuation):
				#	print("Wrong punctuation detected, word {} at {}".format(word, counter))
				#else:
				#	print("Correct punctuation, word {} isPunctuation {}".format(word, punct.isPunctuation()))
	return root
	
def anotateTreeWithBlock(block, regex, listNodes):
	# TODO fix with actual anotation tree read
	posTagRegex, wordRegex = regex
	# print(posTagRegex, wordRegex)
	# Regex shold get group POStag for 1 and word for 2 (2 is applied multiple times)
	prevTag = None
	# must add a position counter to tag correctly
	counter = 0
	for line in block:
		tag = re.search(posTagRegex, line)
		# print("tag {}".format(tag))
		if(tag):
			tag = tag.group(1)
			prevTag = tag
		else:
			tag = prevTag
		match = re.findall(wordRegex, line)
		# print(line, tag, match)
		for trueTag, word in match:
			# print('word {} tag {} counter {}'.format(word, tag, counter))
			node = next((x for x in listNodes[counter:] if x.word == word), None)
			if(node is not None):
				node.addPosTag(trueTag, tag)
				counter += 1
		
	
def getAlignment(line, sentence=None):
	pairing = re.split(' ', line.strip())
	# print(pairing)
	alignmentDict = {}
	matchRegex = re.compile("(\d+)-(\d+)")
	for pair in pairing:
		match = re.match(matchRegex, pair)
		if(match):
			alignmentDict[int(match.group(1))+1] = int(match.group(2))+1
		else:
			# print("Error during getAlignment(no match), string {}".format(pair))
			continue
	return alignmentDict
	
def assignAlignment(alignmentPairs, wordTree):
	# Crude - check all sibling and parent-child relation, and see if they should be switched
	# Choose the switch that is the longest
	allApplicableSwitch = {}
	def getMaxSwitch(node, other):
		# Find all switchable (minus parent) and get the one with maximum distance
		if(node.pos not in alignmentPairs):
			return
		allTargets = []
		allTargets.extend(other)
		allTargets.extend(node.children)
		for target in allTargets:
			if(target is node or target.pos not in alignmentPairs):
				continue
			# print(target.pos, node.pos, alignmentPairs)
			if((target.pos - node.pos) * (alignmentPairs[target.pos] - alignmentPairs[node.pos]) > 0):
				# Correct positioning relative to each other, not switch
				continue
			# Check: if smaller than what is recorded in allApplicableSwitch, change
			# save in (length - target) format
			length, _ = allApplicableSwitch.get(node.pos, (-1, 0))
			otherLength, _ = allApplicableSwitch.get(target.pos, (-1, 0))
			
			if(max(length, otherLength) < abs(target.pos - node.pos)):
				# print("Add new swap: {}-{}".format(node.pos, target.pos))
				# should switch
				allApplicableSwitch[node.pos] = (abs(target.pos - node.pos), target.pos)
				allApplicableSwitch[target.pos] = (abs(target.pos - node.pos), node.pos)
	
	# Check all sibling and parent-child relation, and see if they should be switched
	# Record all switches
	def getAllSwitch(node, other):
		# Find all switchable (minus parent) and save them into key as array
		if(node.pos not in alignmentPairs):
			return
		allTargets = []
		allTargets.extend(other)
		allTargets.extend(node.children)
		allSwitchingPair = []
		for target in allTargets:
			if(target is node or target.pos not in alignmentPairs):
				continue
			# print(target.pos, node.pos, alignmentPairs)
			if((target.pos - node.pos) * (alignmentPairs[target.pos] - alignmentPairs[node.pos]) > 0):
				# Correct positioning relative to each other, not switch
				continue
			if(target.pos < node.pos):
				allSwitchingPair.append((target.pos, node.pos))
			else:
				allSwitchingPair.append((node.pos, target.pos))
		if(len(allSwitchingPair) > 0):
			allApplicableSwitch[node.pos] = allSwitchingPair
	
	def getSiblings(node, child):
		return node.children
	# recursiveRun(getMaxSwitch, wordTree, [], getSiblings)
	recursiveRun(getAllSwitch, wordTree, [], getSiblings)
	
	# Convert back to idx-to-idx dictionary
	# allApplicableSwitch = {k: allApplicableSwitch[k][1] for k in allApplicableSwitch}
	
	return allApplicableSwitch

def alignmentToTrainingExample(wordTree, switchList):
	# training examples will be recorded with parent_child number if available
	examples = {}
	allNode = wordTree.getAllNodeInTree()
	num = len(allNode)
	keyFormat = "{}_{}"
	# Record all examples that is switching
	for switch in switchList:
		if(isinstance(switchList[switch], int)):
			# Longest Switch mode, check along
			key = keyFormat.format(switch, switchList[switch]) #switch + '_' + switchList[switch]
			otherKey = keyFormat.format(switchList[switch], switch) #switchList[switch] + '_' + switch
			if(key in examples or otherKey in examples):
				continue
			node = next((x for x in allNode if x.pos == switch), None)
			other = next((x for x in allNode if x.pos == switchList[switch]), None)
			if(node is None or other is None):
				print("Error @alignmentToTrainingExample, allNode {}, key {}".format(allNode, key))
			front = 'PAC' if(other in node.children or node in other.children) else 'SIB'
			if(front == 'PAC' and node in other.children):
				examples[otherKey] = (front, other, node, True)
			else:
				examples[key] = (front, node, other, True)
		elif(isinstance(switchList[switch], list)):
			# All Switch mode, record all into examples
			# Tuple should be formatted smaller - bigger already
			listCouple = switchList[switch]
			for couple in listCouple:
				key = keyFormat.format(couple[0], couple[1])
				if(key in examples):
					continue
				node = next((x for x in allNode if x.pos == couple[0]), None)
				other = next((x for x in allNode if x.pos == couple[1]), None)
				if(node is None or other is None):
					print("Error @alignmentToTrainingExample, allNode {}, key {}".format(allNode, key))
				front = 'PAC' if(other in node.children or node in other.children) else 'SIB'
				if(front == 'PAC' and node in other.children):
					examples[key] = (front, other, node, True)
				else:
					examples[key] = (front, node, other, True)
			
	
	# Record all examples that available, ignore if recorded (switch)
	def checkAllSwitches(node, siblings):
		# Do not run if node is root
		if(node.word == 'root' and node.pos == 0):
			return
		# Generate sibling training cases
		for other in siblings:
			if(other is node):
				continue
			key = keyFormat.format(node.pos, other.pos) #node.pos + '_' + other.pos
			otherKey = keyFormat.format(other.pos, node.pos) #other.pos + '_' + node.pos
			if(key in examples or otherKey in examples):
				continue
			else:
				examples[key] = ('SIB', node, other, False)
		# Generate parent-child training cases
		for other in node.children:
			# other.addParent(node)
			key = keyFormat.format(node.pos, other.pos) #node.pos + '_' + other.pos
			if(key in examples):
				continue
			else:
				examples[key] = ('PAC', node, other, False)
		
	def getSiblings(node, child):
		return node.children
		
	recursiveRun(checkAllSwitches, wordTree, [], getSiblings)
	
	return examples

def convertExamplesToStrings(examples):
	listOutput = []
	for key in examples:
		type, node, other, switch = examples[key]
		if(node.dependency == WordTree.punctuation_dep or other.dependency == WordTree.punctuation_dep):
			# punctuation node, discarding cases
			continue
		if(type == 'PAC'):
			# Parent-child case, format
			listOutput.append(PACFormatString.format(key, 
				node.word, node.tag, node.dependency, other.word, other.tag, other.dependency,
				getPunctuationInBetween(other, node), getDistance(other, node), 1 if(switch) else 0))
		else:
			# Siblings case, format
			parent = node.parent
			listOutput.append(SIBFormatString.format(key, 
				node.word, node.tag, node.dependency, getDistance(node), 
				other.word, other.tag, other.dependency, getDistance(other),
				parent.word, parent.tag, getPunctuationInBetween(other, None, node), 1 if(switch) else 0))
	return listOutput
	
def writeExamplesToFile(fileout, examples):
	for key in examples:
		type, node, other, switch = examples[key]
		if(node.dependency == WordTree.punctuation_dep or other.dependency == WordTree.punctuation_dep):
			# punctuation node, discarding cases
			continue
		if(type == 'PAC'):
			# Parent-child case, format
			fileout.write(PACFormatString.format(key, 
				node.word, node.tag, node.dependency, other.word, other.tag, other.dependency,
				getPunctuationInBetween(other, node), getDistance(other, node), 1 if(switch) else 0))
		else:
			# Siblings case, format
			parent = node.parent
			fileout.write(SIBFormatString.format(key, 
				node.word, node.tag, node.dependency, getDistance(node), 
				other.word, other.tag, other.dependency, getDistance(other),
				parent.word, parent.tag, getPunctuationInBetween(other, None, node), 1 if(switch) else 0))

def writeStringsToFile(fileOut, strings):
	for str in string:
		fileOut.write(string)

def getDistance(child, parent=None):
	if(parent is None):
		parent=child.parent
	# Distance parent to children
	value = 1
	# Find if any child inbetween, add 1 if do
	childInBetween = next((1 for x in parent.children if ((parent.pos - x.pos) * (x.pos - child.pos)) > 0), 0)
	leftOrRight = -1 if(child.pos < parent.pos) else 1
	return leftOrRight * (1 + childInBetween)
	
def getPunctuationInBetween(child, parent=None, other=None):
	# Try checking punctuation for all existing node instead:
	if(parent is None):
		parent = child.parent
	if(other is None):
		other = parent
	root = parent
	while(hasattr(root, 'parent')):
		root = root.parent
	return next((1 for x in root.getAllNodeInTree() if ((other.pos - x.pos) * (x.pos - child.pos)) > 0 and x.isPunctuation()), 0)
	# Disabled
	if(parent is None):
		parent = child.parent
	if(other is None):
		return next((1 for x in parent.children if ((parent.pos - x.pos) * (x.pos - child.pos)) > 0 and x.isPunctuation()), 0)
	else:
		return next((1 for x in parent.children if ((other.pos - x.pos) * (x.pos - child.pos)) > 0 and x.isPunctuation()), 0)
	
def isNextToEachOther(one, other, parent=None):
	if(parent is None):
		if(one.parent != other.parent):
			raise Exception("Children of different node: {} {}".format(one.word, other.word))
		parent = one.parent
	#if(one.pos > other.pos):
	#	one, other = other, one
	relatedList = list(parent.children)
	relatedList.append(parent)
	return next((1 for x in parent.children if ((other.pos - x.pos) * (x.pos - one.pos)) > 0), 0)

def getTagFromFile(filetagDir, combined=False):
	filetag = io.open(filetagDir, 'r', encoding='utf-8')
	data = filetag.readlines()
	# Seperate
	posTag = {}
	dependencyTag = {}
	currentBlock = None
	tagGetterRegex = "(.+)\t(.+)"
	counter = 0
	for line in data:
		if(line.find('POStag') == 0):
			currentBlock = posTag
			tagCounter = 1
		elif(line.find('dependency') == 0):
			currentBlock = dependencyTag
			tagCounter = 1
		elif(currentBlock is not None):
			# Add line to block
			match = re.match(tagGetterRegex, line)
			if(match is not None):
				currentBlock[match.group(1)] = counter
				counter += 1
	if(combined):
		posTag.update(dependencyTag)
		return posTag
	else:
		return (posTag, dependencyTag)
	
def createWordDictFromFile(filewordDir):
	fileword = io.open(filewordDir, 'r', encoding='utf-8')
	line = fileword.readline()
	wordDict = {}
	# First line specify number of words and dimension size
	# TODO use these data
	line = fileword.readline()
	while line:
		# Format in tokenized word-vector:
		line = line.strip().split()
		vector = []
		wordDict[line[0]] = vector
		for dim in line[1:]:
			try:
				value = float(dim)
			except ValueError:
				value = 0.0
			vector.append(value)
		line = fileword.readline()
	return wordDict
	
def createRefDictFromListTree(treeList):
	refDict = {}
	for tree in treeList:
		for node in tree.getAllNodeInTree():
			refDict[node.word] = refDict.get(node.word, 0) + 1
	return refDict

def createMinimalWordDictFromFile(filewordDir, refDict):
	fileword = io.open(filewordDir, 'r', encoding='utf-8')
	line = fileword.readline()
	wordDict = {}
	# First line specify number of words and dimension size
	# TODO use these data
	line = fileword.readline()
	while line and len(refDict) > 0:
		# Format in tokenized word-vector:
		line = line.strip().split()
		if(line[0] in refDict):
			vector = []
			wordDict[line[0]] = vector
			for dim in line[1:]:
				try:
					value = float(dim)
				except ValueError:
					value = 0.0
				vector.append(value)
			refDict.pop(line[0])
		line = fileword.readline()
	return wordDict
	
def createFormatPAC(separator):
	formatPAC = separator.join(['%s' for i in range(5)])
	def formatter(string, parent, child):
		return string % (parent.tag, parent.dependency, child.tag, child.dependency, getDistance(child))
	return formatPAC, formatter
	
def createFormatSIB(separator):
	formatSIB = separator.join(['%s' for i in range(6)])
	def formatter(string, child, other, parent):
		return string % (parent.tag, child.tag, child.dependency, other.tag, other.dependency, isNextToEachOther(child, other, parent))
	return formatSIB, formatter
	
def addAllTreeRelationToDict(tree, alignmentDict, fullRelationDict, separator="_"):
	formatPAC, formatterPAC = createFormatPAC(separator) # = "PAC" + separator + "%s" + separator + "%s" + separator + "%s"
	def tryAddingAllPAC(node):
		for child in node.children:
			# key = formatPAC % (node.tag, child.dependency, getDistance(child))
			key =  "PAC" + separator + formatterPAC(formatPAC, node, child)
			switch, total = fullRelationDict.get(key, (0, 0))
			total += 1
			if(node.pos in alignmentDict and child.pos in alignmentDict):
				switch += 1 if ((node.pos - child.pos) * (alignmentDict[node.pos] - alignmentDict[child.pos]) < 0) else 0
			fullRelationDict[key] = (switch, total)
	
	formatSIB, formatterSIB = createFormatSIB(separator) # = "SIB" + separator + "%s" + separator + "%s" + separator + "%s" + separator + "%s"
	def tryAddingAllSIB(node):
		listChildren = list(node.children)
		while(len(listChildren) > 1):
			child = listChildren.pop()
			for other in listChildren:
				if(child is other):
					continue
				elif(child.pos > other.pos):
					child, other = other, child
				# key = formatSIB % (node.tag, child.dependency, other.dependency, isNextToEachOther(child, other, node))
				key = "SIB" + separator + formatterSIB(formatSIB, child, other, node)
				switch, total = fullRelationDict.get(key, (0, 0))
				total += 1
				if(other.pos in alignmentDict and child.pos in alignmentDict):
					switch += 1 if ((other.pos - child.pos) * (alignmentDict[other.pos] - alignmentDict[child.pos]) < 0) else 0
				fullRelationDict[key] = (switch, total)
	
	def addAllRelation(node):
		tryAddingAllPAC(node)
		tryAddingAllSIB(node)
	
	recursiveRun(addAllRelation, tree)
	
	return fullRelationDict

def recursiveRun(func, node, other=None, otherFunc=None):
	# Run recursive on a WordTree node
	# print("recursive on node {}".format(node.word))
	if(other is not None):
		func(node, other)
	else:
		func(node)
	for child in node.children:
		if(otherFunc is not None):
			other = otherFunc(node, child)
			recursiveRun(func, child, other, otherFunc)
		else:
			recursiveRun(func, child)

def bottomUpRecursiveRun(func, node, other=None, otherFunc=None):
	# Run bottom-up recursive on a WordTree node
	# print("bottom-up recursive on node {}".format(node.word))
	for child in node.children:
		if(otherFunc is not None):
			other = otherFunc(node, child)
			bottomUpRecursiveRun(func, child, other, otherFunc)
		else:
			bottomUpRecursiveRun(func, child)
	if(other is not None):
		func(node, other)
	else:
		func(node)
	
class WordTree:
	common_spacing = ' '
	punctuation_dep = 'punct'
	listPunctuation = ".,;:!?\"\'`-"
	
	def __init__(self, word, children, pos, dependency):
		# print("Create Node: {} - {}".format( word, pos))
		self.children = children
		self.pos = pos
		self.word = word
		if(dependency.find('_') > 0 or dependency.find(':') > 0 ):
			# Take only first value, for ConLL format
			self.dependency = re.split('[_:]', dependency)[0]
		else:
			self.dependency = dependency
		
		self.tag = 'null'
		
	def getAllNodeInTree(self):
		result = [self]
		for child in self.children:
			result.extend(child.getAllNodeInTree())
		return result
			
	def addPosTag(self, tag, otherTag):
		self.tag = tag
		self.otherTag = otherTag
		
	def addParent(self, parent):
		self.parent = parent
		
	def isPunctuation(self):
		return self.dependency == WordTree.punctuation_dep
		"""self.phrase = word
		 and self.word in WordTree.listPunctuation
		for child in children:
			if(child.pos < self.pos):
				self.phrase = child.phrase + WordTree.common_spacing + self.phrase
			else:
				self.phrase = self.phrase + WordTree.common_spacing + child.phrase"""

connectionRegex = re.compile("([\w:]+)\((.+)-(\d+), (.+)-(\d+)\)")
posTagRegex = re.compile("(?!\(\w+ \w)\((\w+)")
wordRegex = re.compile("\(([\w$]+) (?!\()(.+?)\)")
firstLineRegex = re.compile("Parsing \[.+\]:(.+)")

def runStanfordParser(fileinDir, extension='parser'):
	# Run parser for stanford
	filealignment = io.open(fileinDir + '.align', 'r', encoding='utf-8')
	fileparser = io.open(fileinDir + '.' + extension, 'r', encoding='utf-8')

	dataBlock = getDataBlock(fileparser.readlines(), firstLineRegex, True)
	# First two lines do not contain data, remove
	dataBlock.pop(0)

	dataBlock = filterBlock(dataBlock, firstLineRegex)
	#testBlock, testOther = dataBlock[0]
	#constructTreeFromBlock(testBlock, (connectionRegex, (posTagRegex, wordRegex)), testOther)
	#return
	treeList = [constructTreeFromBlock(block, (connectionRegex, (posTagRegex, wordRegex)), other, line) 
						for block, other, line in dataBlock]

	alignmentList = [getAlignment(line) for line in filealignment.readlines()]
	#for line in dataBlock[0]:
		#print(line)
	# print(testTree.word)
	
	filealignment.close()
	fileparser.close()
	
	return (treeList, alignmentList)
	
conllDataRegex = re.compile("(\d+)\t(.+)\t_\t([\w$\.,:\"\'-]+)\t([\w$\.,:\"\'-]+)\t_\t(\d+)\t(\w+)")
def runConllParser(fileinDir, extension='parser', alignExtension='align'):
	# Run parser for conll
	fileparser = fileinDir if extension is None else fileinDir + '.' + extension
	fileparser = io.open(fileparser, 'r', encoding='utf-8')
	if(alignExtension is not None):
		filealignment = io.open(fileinDir + '.' + alignExtension, 'r', encoding='utf-8')
	
	dataBlock = getDataBlock(fileparser.readlines(), blankLineRegex)
	
	# conll regex, pos - word - POStag - POStag - fatherPos - dependency
	treeList = (constructTreeFromBlock(block, conllDataRegex) for block in dataBlock)
	if(alignExtension is not None):
		alignmentList = [getAlignment(line) for line in filealignment.readlines()]
	
	fileparser.close()
	if(alignExtension is not None):
		filealignment.close()
		return (treeList, alignmentList)
	else:
		return treeList

def defaultKeyFormatter(key):
	return ' '.join(key.split('_'))
	
def writeAllRelationToFile(relationDict, fileDir, splitRule, keyFormatter=defaultKeyFormatter):
	if(splitRule):
		fileoutPAC = io.open(fileDir + '.pac.rule', 'w', encoding='utf-8')
		fileoutSIB = io.open(fileDir + '.sib.rule', 'w', encoding='utf-8')
	else:
		fileout = io.open(fileDir + '.rule', 'w', encoding='utf-8')
	for key in sorted(relationDict):
		if(splitRule):
			fileout = fileoutPAC if("PAC" in key) else fileoutSIB if("SIB" in key) else None
			if(fileout is None):
				# Should have key error here
				continue
		switch, total = relationDict[key]
		percentage = float(switch) / float(total) * 100.0
		key = keyFormatter(key)
		fileout.write(u'{}'.format(key))
		for _ in range(26-len(key)):
			fileout.write(u' ')
		fileout.write(u'%5d/%5d\t(%2.2f%%)\n' % (switch, total, percentage))
	
	if(splitRule):
		fileoutPAC.close()
		fileoutSIB.close()
	else:
		fileout.close()
	
def createRelationData(treeList, alignmentList):
	allRelationDict = {}
	
	for caseIdx in range(len(alignmentList)):
		tree = next(treeList)
		# tree = treeList[caseIdx]
		alignment = alignmentList[caseIdx]
		allRelationDict = addAllTreeRelationToDict(tree, alignment, allRelationDict)
		deleteTree(tree)
	
	return allRelationDict

def createWritingData(treeList, alignmentList, convertToString=False):
	allExamples = []
	
	for caseIdx in range(len(alignmentList)):
		tree = next(treeList)
		alignment = alignmentList[caseIdx]
		switchList = assignAlignment(alignment, tree)
		examples = alignmentToTrainingExample(tree, switchList)
		if(convertToString):
			allExamples.extend(convertExamplesToStrings(examples))
			deleteTree(tree)
			del examples
		else:
			allExamples.append(examples)
	
	return allExamples
	
def writeData(fileoutDir, allExamples):
	fileout = io.open(fileoutDir + '.train', 'w', encoding='utf-8')
	for example in allExamples:
		if(isinstance(example, dict)):
			writeExamplesToFile(fileout, example)
		elif(isinstance(example, str)):
			fileout.write(example)
		else:
			print("Error @writeData: unrecognized type, object type {}".format(example.type))
		fileout.write(u'\n')
	fileout.close()
	
def deleteTree(treeRoot):
	def removeChildren(node):
		del node.children[:]
	
	bottomUpRecursiveRun(removeChildren, treeRoot)
	del treeRoot

def writeDataArff(fileoutDir, allExamples, wordDict, wordLen, tagDict, wordDefaultKey='<unk>'):
	# TODO case for tagDict
	posTag, dependencyTag = tagDict
	wordDefaultKey = wordDict.get(wordDefaultKey, None)
	
	def getWordForVector(wordName, vectorsize):
		return [wordName + 'Vector' + str(i+1) for i in range(vectorsize)]
		
	def getStringFromVector(wordVector, vectorsize=-1):
		if(vectorsize == -1):
			vectorsize = len(wordVector)
		result = ''
		for dim in wordVector:
			result += str(dim) + ", "
		return result
		
	fileoutpac = io.open(fileoutDir + '_pac.arff', 'w', encoding='utf-8')
	fileoutsib = io.open(fileoutDir + '_sib.arff', 'w', encoding='utf-8')
	# Write the class structure for PAC
	fileoutpac.write(u"@RELATION parent-child\n\n")
	fileoutpac.write(u"@ATTRIBUTE parentPos  NUMERIC\n")
	fileoutpac.write(u"@ATTRIBUTE childPos  NUMERIC\n")
	for dim in getWordForVector('parent', wordLen):
		fileoutpac.write(u"@ATTRIBUTE {} NUMERIC\n".format(dim))
	fileoutpac.write(u"@ATTRIBUTE parentPosTagId  NUMERIC\n")
	fileoutpac.write(u"@ATTRIBUTE parentDependencyTagId  NUMERIC\n")
	for dim in getWordForVector('child', wordLen):
		fileoutpac.write(u"@ATTRIBUTE {} NUMERIC\n".format(dim))
	fileoutpac.write(u"@ATTRIBUTE childPosTagId  NUMERIC\n")
	fileoutpac.write(u"@ATTRIBUTE childDependencyTagId  NUMERIC\n")
	fileoutpac.write(u"@ATTRIBUTE punctuationSeperator  NUMERIC\n")
	fileoutpac.write(u"@ATTRIBUTE parentChildDistance  NUMERIC\n")
	
	fileoutpac.write(u"\n@ATTRIBUTE class  {0, 1}\n")
	fileoutpac.write(u"\n@data\n")
	
	# Write the class structure for SIB
	fileoutsib.write(u"@RELATION siblings\n\n")
	fileoutsib.write(u"@ATTRIBUTE onePos  NUMERIC\n")
	fileoutsib.write(u"@ATTRIBUTE otherPos  NUMERIC\n")
	for dim in getWordForVector('one', wordLen):
		fileoutsib.write(u"@ATTRIBUTE {} NUMERIC\n".format(dim))
	fileoutsib.write(u"@ATTRIBUTE onePosTagId  NUMERIC\n")
	fileoutsib.write(u"@ATTRIBUTE oneDependencyTagId  NUMERIC\n")
	fileoutsib.write(u"@ATTRIBUTE oneParentDistance  NUMERIC\n")
	for dim in getWordForVector('other', wordLen):
		fileoutsib.write(u"@ATTRIBUTE {} NUMERIC\n".format(dim))
	fileoutsib.write(u"@ATTRIBUTE otherPosTagId  NUMERIC\n")
	fileoutsib.write(u"@ATTRIBUTE otherDependencyTagId  NUMERIC\n")
	fileoutsib.write(u"@ATTRIBUTE otherParentDistance  NUMERIC\n")
	for dim in getWordForVector('parent', wordLen):
		fileoutsib.write(u"@ATTRIBUTE {} NUMERIC\n".format(dim))
	fileoutsib.write(u"@ATTRIBUTE parentPosTagId  NUMERIC\n")
	fileoutsib.write(u"@ATTRIBUTE punctuationSeperator  NUMERIC\n")
	
	fileoutsib.write(u"\n@ATTRIBUTE class  {0, 1}\n")
	
	fileoutsib.write(u"\n@data\n")
	
	
	# Write the data into respective file
	for example in allExamples:
		for key in example:
			type, node, other, switch = example[key]
			# Ignore if punctuation key
			if(node.dependency == WordTree.punctuation_dep or other.dependency == WordTree.punctuation_dep):
				# punctuation node, discarding cases
				continue
			if(type == 'PAC'):
				# Word will have trailing comma, so it cannot have comma behind it
				parentVectorString = getStringFromVector(wordDict.get(node.word, wordDefaultKey))
				childVectorString = getStringFromVector(wordDict.get(other.word, wordDefaultKey))
				# parentPos, childPos, parentVector, parentPosTagId, parentDependencyTagId,
				# childVector, childPosTagId, childDependencyTagId, punctuation, distance, class
				try:
					fileoutpac.write("{}, {}, {} {}, {}, {} {}, {}, {}, {}, {}\n".format(node.pos, other.pos,
						parentVectorString, posTag[node.tag], dependencyTag[node.dependency],
						childVectorString, posTag[other.tag], dependencyTag[other.dependency],
						getPunctuationInBetween(other, node), getDistance(other, node), 1 if(switch) else 0))
				except KeyError:
					print("KeyError, list: {} {} {} {}".format(node.tag, node.dependency, other.tag, other.dependency))
					print("KeyError, exist: {} {} {} {}".format(node.tag in posTag, node.dependency in dependencyTag, other.tag in posTag, other.dependency in dependencyTag))
			elif(type == 'SIB'):
				# Sibling cases should have the left one first, hence switch
				if(node.pos > other.pos):
					node, other = other, node
				# Word will have trailing comma, so it cannot have comma behind it
				parent = node.parent
				# Remove if parent is punctuation
				#if(parent.dependency == WordTree.punctuation_dep):
					# punctuation node, discarding cases
				#	continue
				parentVectorString = getStringFromVector(wordDict.get(parent.word, wordDefaultKey))
				oneVectorString = getStringFromVector(wordDict.get(node.word, wordDefaultKey))
				otherVectorString = getStringFromVector(wordDict.get(other.word, wordDefaultKey))
				# onePos, otherPos, oneVector, onePosTagId, oneDependencyTagId, oneParentDistance,
				# otherVector, otherPosTagId, otherDependencyTagId, otherParentDistance,
				# parentVector, parentPosTagId, class
				try:
					fileoutsib.write("{}, {}, {} {}, {}, {}, {} {}, {}, {}, {} {}, {}, {}\n".format(node.pos, other.pos,
						oneVectorString, posTag[node.tag], dependencyTag[node.dependency], getDistance(node),
						otherVectorString, posTag[other.tag], dependencyTag[other.dependency], getDistance(other),
						parentVectorString, posTag[parent.tag],
						getPunctuationInBetween(other, node), 1 if(switch) else 0))
				except KeyError:
					print("KeyError, list: {} {} {} {} {}".format(node.tag, node.dependency, other.tag, other.dependency, parent.tag))
					print("KeyError, exist: {} {} {} {} {}".format(node.tag in posTag, node.dependency in dependencyTag, other.tag in posTag, other.dependency in dependencyTag, parent.tag in posTag))
			
	
	fileoutpac.close()
	fileoutsib.close()

if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Create training examples from resource data.')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input files')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-t','--tagdir', type=str, default="all_tag.txt", help='location of the tag file containing both POStag and dependency, default all_tag.txt')
	parser.add_argument('-m','--mode', type=str, default="conll", help='parser file mode (stanford|conll), default conll')
	parser.add_argument('-x','--output_extension', type=str, default="txt", help='output file mode (txt|arff|rule), default txt')
	parser.add_argument('-e','--embedding_word', type=str, help='embedding word file')
	parser.add_argument('--parser_extension', type=str, default="parser", help='file extension for parser, default parser')
	parser.add_argument('--default_word', type=str, default="*UNKNOWN*", help='key in embedding_word standing for unknown word. Only in arff mode')
	parser.add_argument('--full_load', action='store_true', help='get all vector for words in embedding_word file. Not recommended for large word file. Only in arff mode')
	parser.add_argument('--min_cases', type=int, default=-1, help='number of cases needed to be recorded, default -1(all), only in rule mode')
	parser.add_argument('--string_mode', action='store_true', help='try to convert to string beforehand, only usable in txt/rule mode to reduce memory usage')
	parser.add_argument('--split_rule', action='store_true', help='split the rule file into PAC/SIB file, rule mode only')
	args = parser.parse_args()
	# args.tagdir = "all_tag.txt"
	# args.embedding_word = "data\syntacticEmbeddings\skipdep_embeddings.txt"
	# args.default_word = "*UNKNOWN*"
	if(args.outputdir is None):
		args.outputdir = args.inputdir
	
	timer = time.time()
	if(args.mode == 'stanford'):
		treeList, alignmentList = runStanfordParser(args.inputdir, args.parser_extension)
		print("Done for @StanfordParser, time passed %.2fs" % (time.time() - timer))
	elif(args.mode == 'conll'):
		treeList, alignmentList = runConllParser(args.inputdir, args.parser_extension)
		print("Done for @ConllParser, time passed %.2fs" % (time.time() - timer))
	else:
		raise argparse.ArgumentTypeError("Incorect mode, must be stanford|conll")
	
	if(args.output_extension == 'txt'):
		allExamples = createWritingData(treeList, alignmentList, args.string_mode)
		print("Done for @createWritingData, time passed %.2fs" % (time.time() - timer))
		writeData(args.outputdir, allExamples)
		print("Done for @writeData, time passed %.2fs" % (time.time() - timer))
	elif(args.output_extension == 'arff'):
		allExamples = createWritingData(treeList, alignmentList, False)
		print("Done for @createWritingData, time passed %.2fs" % (time.time() - timer))
		if(not args.full_load):
			# Doing minimal (not load full wordDict)
			refDict = createRefDictFromListTree(treeList)
			refDict[args.default_word] = 1
			wordDict = createMinimalWordDictFromFile(args.embedding_word, refDict)
			print("Done for @creatingWordDict (minimal), time passed %.2fs" % (time.time() - timer))
		else:
			# Doing full load, not recommended for large file
			wordDict = createWordDictFromFile(args.embedding_word)
			print("Done for @creatingWordDict (full), time passed %.2fs" % (time.time() - timer))
		tagDict = getTagFromFile(args.tagdir)
		# WordLen is a random key's vector length
		wordLen = len(wordDict[args.default_word])
		print("Done for @creatingTagDict, time passed %.2fs" % (time.time() - timer))
		writeDataArff(args.outputdir, allExamples, wordDict, wordLen, tagDict, args.default_word)
		print("Done for @writeDataArff, time passed %.2fs" % (time.time() - timer))
	elif(args.output_extension == 'rule'):
		allRelation = createRelationData(treeList, alignmentList)
		allRelation = dict((k, v) for k, v in allRelation.items() if v[1] >= args.min_cases)
		print("Done for @createRelationData, time passed %.2fs" % (time.time() - timer))
		writeAllRelationToFile(allRelation, args.outputdir, args.split_rule)
		print("Done for @writeAllRelationToFile, time passed %.2fs" % (time.time() - timer))
	else:
		raise argparse.ArgumentTypeError("Incorect output_extension, must be txt|arff|rule")
	# Execute script