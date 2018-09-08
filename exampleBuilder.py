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

def createFormatModel(numChild, separator):
	numTags = 2 * numChild + 1
	modelString = ["{}"] * numTags
	modelString = separator.join(modelString)
	# print("modelString: " + modelString)
	
	def modelFunction(formatString, node, listChild=None):
		listChild = listChild or node.children
		#allTags = [node.tag] + [child.tag for child in node.children] + [child.dependency for child in node.children]
		allTags = [node.tag] + [f(child) for child in listChild for f in (lambda x: x.tag, lambda x:x.dependency)]
		assert len(allTags) == numTags, "allTags: {} != numTags {}".format(allTags, numTags)
		return formatString.format(*allTags)
		
	def modelValue(node, alignment, listChild=None):
		listChild = listChild or node.children
		norminal_positioned = [node] + listChild
		old_positioned = sorted(norminal_positioned, key=lambda x: x.pos)
		align_positioned = sorted(norminal_positioned, key=lambda x: alignment[x.pos] if(x.pos in alignment) else x.pos)
		
		key = separator.join([str(norminal_positioned.index(n)) for n in old_positioned])
		value = separator.join([str(norminal_positioned.index(n)) for n in align_positioned])
		return key + '>' + value
	
	return modelString, modelFunction, modelValue
	
def splitChildrenByPunctuation(children):
	if(any(x.isPunctuation() for x in children) and len(children) > 1):
		punct_indexes = [i for i,x in enumerate(children) if x.isPunctuation()]
		if(0 not in punct_indexes):
			punct_indexes = [-1] + punct_indexes
		# Create slicing range based on the indexes of the punctuation, excluding these punctuations themselves
		punct_slicing = [(punct_indexes[i]+1, punct_indexes[i+1] if i+1<len(punct_indexes) else len(children)) for i in range(len(punct_indexes))]
		splitted_list = [children[start:end] for start, end in punct_slicing if start < end]
		return splitted_list
	else:
		return [children]

def addTreeRelationByModelType(numChild, tree, alignmentDict, fullRelationDict, splitByPunctuation=False, separator = '_'):
	modelString, modelKeyFunction, modelValue= createFormatModel(numChild, separator) # = "PAC" + separator + "%s" + separator + "%s" + separator + "%s"
	def tryAddingAllForMode(node):
		if(splitByPunctuation):
			listChildGroup = splitChildrenByPunctuation(node.children)
		else:
			listChildGroup = [node.children]
		for listChild in listChildGroup:
			if(len(listChild) == numChild):
				# key = formatPAC % (node.tag, child.dependency, getDistance(child))
				key =  modelKeyFunction(modelString, node, listChild)
				posKey = modelValue(node, alignmentDict, listChild)
				recorder = fullRelationDict.get(key, {})
				fullRelationDict[key] = recorder
				recorder[posKey] = recorder.get(posKey, 0) + 1
	
	recursiveRun(tryAddingAllForMode, tree)
	
	return fullRelationDict
	
def createRelationDataByModelType(numChild, treeList, alignmentList, splitByPunctuation=False):
	allRelationDict = {}
	
	for caseIdx in range(len(alignmentList)):
		tree = next(treeList)
		# tree = treeList[caseIdx]
		alignment = alignmentList[caseIdx]
		allRelationDict = addTreeRelationByModelType(numChild, tree, alignment, allRelationDict, splitByPunctuation=splitByPunctuation)
		deleteTree(tree)
	
	return allRelationDict
	

def recordDepTagRelation(tree, alignmentDict, fullRelationDict, separator='_'):
	depString = separator.join(["{}"] * 2)
	depKeyFunc = lambda x1, x2: depString.format(x1,x2)
	depSelfKey = "self"
	def depTagInNode(node):
		depTagRespectiveDict = fullRelationDict.get(node.tag, {})
		fullRelationDict[node.tag] = depTagRespectiveDict
		# Do a more readable traversal
		traverseItem = [node] + node.children
		while(len(traverseItem) > 1):
			firstNode = traverseItem[0]
			traverseItem = traverseItem[1:]
			for secondNode in traverseItem:
				firstDep, secondDep = firstNode.dependency if firstNode is not node else depSelfKey, secondNode.dependency if secondNode is not node else depSelfKey
				firstTruePos, secondTruePos = alignmentDict.get(firstNode.pos, 0), alignmentDict.get(secondNode.pos, 0)
				isSwap = (firstTruePos - secondTruePos) * (firstNode.pos - secondNode.pos) < 0
				if(firstDep > secondDep):
					# Reverse the tags and values as needed
					firstDep, secondDep = secondDep, firstDep
					firstTruePos, secondTruePos = secondTruePos, firstTruePos
				isLeft = firstTruePos < secondTruePos
				key = depKeyFunc(firstDep, secondDep)
				# the recorded values are tuples of (first) isLeft (second), (first) isRight (second), isSwap, total
				isLeftScore, isRightScore, isSwapScore, totalScore = depTagRespectiveDict.get(key, (0,0,0,0))
				depTagRespectiveDict[key] = (isLeftScore + int(isLeft), isRightScore + int(not isLeft), isSwapScore + int(isSwap), totalScore+1) 
	# execute the previous function
	recursiveRun(depTagInNode, tree)
	
	return fullRelationDict

def createDepRelationByModelType(treeList, alignmentList, bracketPreferedSize=5, separator='_', debug=False, injectionMode=False, bruteforceMode=False):
	allRelationDict = {}
	
	for caseIdx in range(len(alignmentList)):
		tree = next(treeList)
		# tree = treeList[caseIdx]
		alignment = alignmentList[caseIdx]
		allRelationDict = recordDepTagRelation(tree, alignment, allRelationDict, separator=separator)
		deleteTree(tree)
	
	depBestRelationDict = {}
	if(injectionMode):
		processRelationFunc = processRelationDictByInjective
	elif(bruteforceMode):
		processRelationFunc = bruteForceRelationDict
	else:
		processRelationFunc = processRelationDict
	
	for key in allRelationDict:
		# depBestRelationDict[key] = processRelationDict(allRelationDict[key], bracketPreferedSize=bracketPreferedSize, separator=separator, debug=debug, POSTag=key)
		depBestRelationDict[key] = processRelationFunc(allRelationDict[key], bracketPreferedSize=bracketPreferedSize, separator=separator, debug=debug, POSTag=key)
		# may not be necessary, but can free up extra memory for later actions. Default not needed
		#del allRelationDict[key]
		
	return depBestRelationDict
	
def bruteForceRelationDict(depDict, maxBracket=10, separator='_', debug=False, POSTag=None, bracketPreferedSize=None):
	# Create every possible combination for each number of brackets, and save the one with the best result
	# As per the name, this will take a retarded amount of time. given we have 20~40 tags, we are looking at 5^20 iterations.
	raise Exception("@bruteForceRelationDict should not be used at all.")
	
	# A small hack to wire the bracketPreferedSize into overriding maxBracket
	maxBracket = bracketPreferedSize if bracketPreferedSize is not None else maxBracket

	# First, list all the available tags inside the depDict
	keyList = depDict.keys()
	tagList = [key for tagChunk in keyList for key in tagChunk.split(separator)]
	# filter tag to single items here
	tagList = list(set(tagList))
	# you cannot have more brackets than tag
	maxBracket = min(maxBracket, len(tagList))
	# exit if blank
	if(len(tagList) <= 1):
		return [tagList], writeScoreGrouped(tagList, depDict, separator) if debug else None
	
	# we will use writeScoreGrouped for scoring purpose, which leaves us with the need to construct all possible iterations of brackets

	firstBracket = [list(tagList)]
	bestScore, bestForm = writeScoreGrouped(firstBracket, depDict, separator)[0], firstBracket

	# function: convert iterNum to respective pos for each tag
	def iterToPos(iterNum, bracket, tagNum):
		return [int(iterNum / (bracket ** idx)) % bracket for idx in range(tagNum)]
	# function: from the pos above, create the corresponding bracket format
	def posToDistrib(pos, tags, bracket):
		distrib = [[]] * bracket
		for tag, tagPos in zip(tags, pos):
			distrib[tagPos].append(tag)
		return distrib

	for bracketSize in range(2, maxBracket+1):
		if(debug):
			localBestScore, localBestDistribution = -1, None
		# iterations are represented as tag_bracket * (tag_id^brackets), eg: tag1 tag2 tag3 in [[tag1 tag3] [tag2]] will have position 0 1 0 and hence equal to values 0 * 2^2 + 1 * 2^1 + 0 * 2^0 = 2
		tagNum = len(tagList)
		iterationSize = bracketSize ** tagNum
		if(debug):
			print("Num of iters for tag {:s} - bracket {:d}: {:d}".format(POSTag, bracketSize, iterationSize))
	
		for iterNum in range(iterationSize):
			tagPos = iterToPos(iterNum, bracketSize, tagNum)
			# filter out those which has empty bracket
			if(any(bracket not in tagPos for bracket in range(bracketSize))):
				continue
			distribution = posToDistrib(tagPos, tagList, bracketSize)
			score = writeScoreGrouped(distribution, depDict, separator)[0]
			if(score >= bestScore):
				del bestForm
				bestScore, bestForm = score, distribution
			
			if(debug and score > localBestScore):
				localBestScore, localBestDistribution = score, distribution
			else:
				del distribution
		# debug print the best of this particular bracket found by brute force
		if(debug):
			print("Best for bracket {:d}: {} (score {:d})".format(bracketSize, localBestDistribution, localBestScore))
	
	# Output the result from the bruteforce
	return bestForm, writeScoreGrouped(bestForm) if debug else None

def processRelationDict(depDict, bracketPreferedSize=5, separator='_', debug=False, POSTag=None):
	# function will be used for the depTagRespectiveDict inside the allRelationDict to create a best order of items in the depDict
	# TODO the valid/total value had been given by debug, but it is only for NORMAL grouping. It works fine now, but it is dangerous for 
	
	# First, list all the available tags inside the depDict
	keyList = depDict.keys()
	tagList = [key for tagChunk in keyList for key in tagChunk.split(separator)]
	# filter tag to single items here
	tagList = list(set(tagList))
	
	# Condense the values for each tag by isLeft - isRight, and sort them by smallest to highest. By default, we count an isLeft as -1 and isRight as 1 for the purpose of sorting
	tagScore = []
	for tag in tagList:
		isLeft, isRight = 0, 0
		relatedKey = [key for key in keyList if tag in key]
		for key in relatedKey:
			if(key.find(tag) == 0):
				increaseLeft, increaseRight, _, _= depDict[key]
			else:
				increaseRight, increaseLeft, _, _ = depDict[key]
			isLeft, isRight = isLeft + increaseLeft, isRight + increaseRight
		tagScore.append(isRight - isLeft)
	# sort the tagList by the previous score, and get them into single-sized list for next step
	tagList = sorted(zip(tagScore, tagList))
	tagList = [item[1] for item in tagList]
	if(debug):
		correctCases, totalCases = writeScoreSimple(tagList, depDict, separator=separator)
		percentageValue = 0.0 if totalCases == 0 else float(correctCases) / float(totalCases) * 100.0
		print("Raw ordering of tag list done, tag {}, result in {}/{} correct cases ({:.2f}%)".format(POSTag, correctCases, totalCases, percentageValue))
	tagList = [[item] for item in tagList]
	
	depString = separator.join(["{}"] * 2)
	# see if any tag chunk is better being grouped in normal/reverse with a neighbor tag chunk
	def checkNeighbor(currentChunk, neighborChunk):
		reverseCases, currentCases, totalCases = 0, 0, 0
		for leftTag in currentChunk:
			for rightTag in neighborChunk:
				depKey, correctOrder = (depString.format(leftTag, rightTag), True) if leftTag < rightTag else (depString.format(rightTag, leftTag), False)
				if(depKey not in depDict):
					continue
#					print("Key {} not found in depDict".format(depKey))
#				elif(debug):
#					print("Key {} found in depDict, value: {}".format(depKey, depDict[depKey]))
				increaseCurrentCases = depDict[depKey][0 if correctOrder else 1]
				increaseReverseCases = depDict[depKey][2]
				increaseTotalCases = depDict[depKey][3]
				reverseCases += increaseReverseCases
				currentCases += increaseCurrentCases
				totalCases += increaseTotalCases
		normalCases = totalCases - reverseCases
#		if(debug):
#			print("Chunk {} - {} have values: current {}, normal {}, reverse {}, total {}".format(currentChunk, neighborChunk,currentCases, normalCases, reverseCases, totalCases))
		if(normalCases > currentCases):
			return True, "NORMAL", normalCases - currentCases
		elif(reverseCases > currentCases):
			return True, "REVERSE", reverseCases - currentCases
		elif(normalCases > reverseCases):
			return False, "NORMAL", currentCases - normalCases
		else:
			return False, "REVERSE", currentCases - reverseCases
	# do exhaustive joining of list, until the list is reduced to a certain size or the reduction will cause a drop in values instead
	while(len(tagList) > bracketPreferedSize):
		connectionList = [checkNeighbor(tagList[i], tagList[i+1]) for i in range(len(tagList)-1)]
		assert len(connectionList) == len(tagList) - 1
		maxCollapsible = max(connectionList, key=lambda item: -item[2] if item[0] is False else item[2])
		if(maxCollapsible[0] is False):
			break
		# with item, reduce tagList by its index
		reductionIdx = connectionList.index(maxCollapsible)
		assert reductionIdx < len(tagList), "reduction item {}-{} in {}, tagList {}".format(reductionIdx, maxCollapsible, connectionList, tagList)
			# pop remove an item, so secondChunk MUST be first to be popped. Learned this the hard way today
		secondChunk = tagList.pop(reductionIdx+1)
		firstChunk = tagList.pop(reductionIdx)
		tagList.insert(reductionIdx, firstChunk + secondChunk)
		if(debug):
			print("Merge group {} and group {} with key {}, correct cases increased: {}".format(firstChunk, secondChunk, maxCollapsible[1], maxCollapsible[2]))
	if(debug):
		return tagList, writeScoreGrouped(tagList, depDict, separator)
	else:
		return tagList, None

def processRelationDictByInjective(depDict, bracketPreferedSize=5, separator='_', debug=False, POSTag=None, detailedDebug=False):
	# Do injective processing: self is automatically 0, and inject other tags into the list by position where it best serve them
	
	# First, list all the available tags inside the depDict
	keyList = depDict.keys()
	tagList = [key for tagChunk in keyList for key in tagChunk.split(separator)]
	# filter tag to single items here
	tagList = list(set(tagList))
	if(len(tagList) <= 1):
		# Should not have 1, but just in case. Eject immediately
		return [tagList], writeScoreGrouped(tagList, depDict, separator) if debug else None
	
	# remove tag self and put it into inject list
	tagList.remove('self')
	injectList = [['self']]
	
	depString = separator.join(["{}"] * 2)
	# function: check if injecting tag into gList at idx will create how many correct cases
	def injectTagToList(tag, gList, idx):
		total = 0
		for gIdx, group in enumerate(gList):
			# is tag left or right of group at this idx
			isLeft = gIdx >= idx
			for groupTag in group:
				if(tag < groupTag):
					dictKey = depString.format(tag, groupTag)
					# depDict[dictKey] is tuple of (1 isLeft 2, 1 isRight 2, 1 swap 2, total)
					isLeftIdx = 0
				else:
					dictKey = depString.format(groupTag, tag)
					isLeftIdx = 1
				# if this injection is the left of the group, use isLeftIdx, else use isRightIdx (1 - isLeftIdx)
				correctCasesrrectIdxForDictValue = isLeftIdx if isLeft else 1 - isLeftIdx
				total += depDict.get(dictKey, (0,0,0,0))[correctCasesrrectIdxForDictValue]
		return total
	# function: check if put tag in existing group (idx) will create how many correct cases (separate between REVERSE and NORMAL)
	def groupTagIntoList(tag, gList, idx):
			totalOther = 0
			normalInc, reverseInc = 0, 0
			for gIdx, group in enumerate(gList):
				if(gIdx == idx):
					for groupTag in group:
						# we don't need to care of who is first in this case
						dictKey = depString.format(tag, groupTag) if tag < groupTag else depString.format(groupTag, tag)
						_, _, reverseValue, totalValue = depDict.get(dictKey, (0,0,0,0))
						normalValue = totalValue - reverseValue
						normalInc += normalValue
						reverseInc += reverseValue
				else:
					# handle exactly like injectTag
					# TODO we can maybe merge it together with float idxs
					isLeft = gIdx >= idx
					for groupTag in group:
						if(tag < groupTag):
							dictKey = depString.format(tag, groupTag)
							# depDict[dictKey] is tuple of (1 isLeft 2, 1 isRight 2, 1 swap 2, total)
							isLeftIdx = 0
						else:
							dictKey = depString.format(groupTag, tag)
							isLeftIdx = 1
						# if this injection is the left of the group, use isLeftIdx, else use isRightIdx (1 - isLeftIdx)
						correctCasesrrectIdxForDictValue = isLeftIdx if isLeft else 1 - isLeftIdx
						totalOther += depDict.get(dictKey, (0,0,0,0))[correctCasesrrectIdxForDictValue]
			# choose NORMAL or REVERSE based on recorded values
			if(normalInc >= reverseInc):
				return totalOther + normalInc, "NORMAL"
			else:
				return totalOther + reverseInc, "REVERSE"

	# function: for every remaining tag in sourceList, try to inject it into targetList and rate for best number of cases raised. output a list of (caseRaised, tag, injectIdx, injectType) per each tag. injectType being NEW (new group), NORMAL/REVERSE (into existing group)
	def injectChecker(sourceList, targetList):
		resultList = []
		for tag in sourceList:
			# Try inject beside every possible group, and get the best pos based on that
			injectList = [(injectTagToList(tag, targetList, pos), pos) for pos in range(len(targetList) + 1)]
			bestInjectVal, bestInjectPos = max(injectList, key=lambda item: item[0])
			# Try group with every group and likewise search for best pos
			# hack to create size3 tuple from size2 one
			groupList = [list(groupTagIntoList(tag, targetList, pos)) + [pos] for pos in range(len(targetList))]
			bestGroupVal, bestGroupingType, bestGroupPos = max(groupList, key=lambda item: item[0])
			if(debug and detailedDebug):
				print("Tag {:s} into {}: INJECT at {:d}(+{:d}), GROUP {:s} at {:d}(+{:d})".format(tag, targetList, bestInjectPos, bestInjectVal, bestGroupingType, bestGroupPos, bestGroupVal))
			if(bestInjectVal > bestGroupVal or (bestInjectVal==bestGroupVal and len(targetList) < bracketPreferedSize)):
				# small favor of injection over group
				resultList.append((bestInjectVal, tag, bestInjectPos, 'NEW'))
			else:
				resultList.append((bestGroupVal, tag, bestGroupPos, bestGroupingType))
		return resultList
		
	if(debug and POSTag is not None):
		print("=== Injecting for parent POSTag {:s} ===".format(POSTag))
	# run a while loop that will inject all from tagList to injectList
	while(len(tagList) > 0):
		allInjections = injectChecker(tagList, injectList)
		injectionVal, injectionTag, injectionPos, injectAction = max(allInjections, key=lambda item: item[0])
		# remove the best tag from tag list and add it to injectList
		tagList.remove(injectionTag)
		if(injectAction == 'NEW'):
			if(debug): 
				print("Best injection: tag {:s} INJECT to pos {:d} increase correct cases by {:d}".format(injectionTag, injectionPos, injectionVal))
			injectList.insert(injectionPos, [injectionTag])
		else:
			if(debug):
				print("Best injection: tag {:s} GROUP with {} by {:s} increase correct cases by {:d}".format(injectionTag, injectList[injectionPos], injectAction, injectionVal))
			injectList[injectionPos].append(injectionTag)
	if(debug):
		return injectList, writeScoreGrouped(injectList, depDict, separator)
	else:
		return injectList, None

def writeScoreSimple(tagList, relationDict, separator='_'):
	# Write score for the initial ordering
	correctCases, totalCases = 0, 0
	for relation in relationDict:
		leftTag, rightTag = relation.split(separator)
		leftTag, rightTag = tagList.index(leftTag), tagList.index(rightTag)
		assert leftTag >= 0 and rightTag >=0
		correctIdx = 0 if leftTag < rightTag else 1
		totalCases += relationDict[relation][3]
		correctCases += relationDict[relation][correctIdx]
	return correctCases, totalCases

def writeScoreGrouped(tagList, relationDict, separator='_'):
	# Write score for the later grouping result
	correctCases, totalCases = 0, 0
	#fullTagList = [tag for groupedTag in tagList for tag in groupedTag]
	for groupIdx, groupedTag in enumerate(tagList):
		for currentTag in groupedTag:
			# Compare with every other tag in tagList
			for comparingIdx, comparingGroup in enumerate(tagList):
				for otherTag in comparingGroup:
					# This sure is getting ugly...
					if(currentTag == otherTag):
						continue
					if(groupIdx < comparingIdx):
						correctCaseIdx = 0
					elif(comparingIdx < groupIdx):
						correctCaseIdx = 1
					else:
						correctCaseIdx = 2
					# reverse the correctCaseIdx in case of not in same group and otherTag<currentTag
					if(correctCaseIdx < 2 and otherTag < currentTag):
						correctCaseIdx = 1 - correctCaseIdx
					# add the correct/total case into the full value if found
					leftTag, rightTag = (currentTag, otherTag) if(currentTag < otherTag) else (otherTag, currentTag)
					relatedKey = leftTag + separator + rightTag
					relation = relationDict.get(relatedKey, (0,0,0,0))
					# as key @2 is isSwap(REVERSE), we must do it with total-isSwap instead
					correctCases += relation[correctCaseIdx] if(correctCaseIdx < 2) else (relation[3] - relation[correctCaseIdx])
					totalCases += relation[3]
	# all relation must be searched through twice, thus assertment must be run as additional constraint
	assert correctCases % 2 == 0 and totalCases % 2 == 0, "Something wrong with the checking logic. (Doubled) correctCases {}, totalCases {}".format(correctCases, totalCases)
	correctCases, totalCases = correctCases / 2, totalCases / 2
	return int(correctCases), int(totalCases)


def writeDepRelationToFile(fullRelationDict, outputdir, writeSeparator="->", debug=False, sortKey=True):
	openedFile = io.open(outputdir + ".dep", "w", encoding='utf-8')
	keyList = fullRelationDict.keys()
	if(sortKey):
		keyList = sorted(keyList)
	for key in keyList:
		depList, statistic = fullRelationDict[key]
		#print(depList, statistic)
		depList = ["(" + " ".join(item) + ")" for item in depList]
		openedFile.write("{}: {}\n".format(key, writeSeparator.join(depList)))
		if(debug and statistic is not None):
			openedFile.write("Valid/Total case for key {} using this particular grouping: {:d}/{:d}({:.2f}%)\n".format(key, statistic[0], statistic[1], float(statistic[0]) / float(statistic[1]) * 100.0 if statistic[1] > 0 else 0.0))
	openedFile.close()

def fillWordToSize(word, size):
	if(len(word) < size):
		return word + ' ' * (size - len(word))
	return word

def writerFormatter(separator='_'):
	def labelWriter(numChild):
		return ' '.join(["_P(T)_"] + ["C{}(T) C{}(dep)".format(i, i) for i in range(1, numChild+1)])
		
	def unpackTags(string):
		tagList = [ fillWordToSize(tag, 5 if(i%2!=0) else 7) for i, tag in enumerate(string.split(separator))]
		return ' '.join(tagList)
	
	def unpackValues(values):
		posBefore, posAfter = values.split('>')
		return posBefore.strip(), posAfter.strip()
	
	return labelWriter, unpackTags, unpackValues

def writeAllModelRelationToFile(numChild, outputdir, fullRelationDict, writePercentage=False, writeModelPercentageValue=None, writeModelThreshold=None):
	openedFile = io.open(outputdir + ".model" + str(numChild), "w", encoding='utf-8')
	labelWriter, unpackTags, unpackValues = writerFormatter()
	openedFile.write(labelWriter(numChild) + '\n')
	for key in fullRelationDict:
		formattedKey = unpackTags(key)
		subtitutionSpace = ' ' * len(formattedKey)
		mutationDict = fullRelationDict[key]
		writeIntoFile = ""
		if(writePercentage or writeModelPercentageValue):
			# Record the overall values and respective case (from/to) in a dict
			totalDict = {}, {}
			totalAppearance = 0
			if(writeModelPercentageValue):
				maxRecorder = {}
			for mutation in mutationDict:
				occurence = mutationDict[mutation]
				startKey, endKey = unpackValues(mutation)
				totalDict[0][startKey] = totalDict[0].get(startKey, 0) + occurence
				totalDict[1][endKey] = totalDict[1].get(endKey, 0) + occurence
				if(writeModelPercentageValue):
					# Check and keep the largest endKey for its startKey
					if(maxRecorder.get(startKey, (0, ""))[0] < occurence):
						maxRecorder[startKey] = (occurence, endKey)
				totalAppearance += occurence
			totalDict = totalDict[0], totalDict[1], float(totalAppearance)
			if(writeModelPercentageValue):
				for startKey in maxRecorder:
					highestVal, highestEndKey = maxRecorder[startKey]
					totalOccurence = totalDict[0][startKey]
					if(float(highestVal) / float(totalOccurence) >= writeModelPercentageValue and startKey != highestEndKey and (writeModelThreshold is None or totalOccurence >= writeModelThreshold[numChild-1])):
						openedFile.write(formattedKey + " {}-{} {}/{}".format(startKey, highestEndKey, highestVal, totalOccurence) + '\n')
				# skip the lower write in this mode
				continue
		else:
			totalDict = None
		for i, mutation in enumerate(mutationDict):
			occurence = mutationDict[mutation]
			startKey, endKey = unpackValues(mutation)
			writeIntoFile += (formattedKey if(i==0) else subtitutionSpace) + " :" + "{} -> {} : ".format(startKey, endKey) + str(occurence)
			if(writePercentage and totalDict is not None):
				occurence = float(occurence)
				startPercentage, endPercentage, totalPercentage = occurence/float(totalDict[0][startKey]), occurence/float(totalDict[1][endKey]), occurence/float(totalDict[2])
				writeIntoFile += "\t%.2f%%(begin) %.2f%%(end) %.2f%%(total)" % (startPercentage * 100.0, endPercentage * 100.0, totalPercentage * 100.0)
			writeIntoFile += '\n'
#		if(totalDict): writeIntoFile += subtitutionSpace + "Total: " + str(totalDict[2])
		openedFile.write(writeIntoFile)
				# "{} -> {} : ".format(*unpackValues(mutation))
		if(totalDict):
			del totalDict
	openedFile.close()
	
def createIterableByMode(mode, inputdir, extension, timer):
	if(mode == 'stanford'):
		treeList, alignmentList = runStanfordParser(inputdir, extension)
		print("Done for @StanfordParser, time passed %.2fs" % (time.time() - timer))
	elif(mode == 'conll'):
		treeList, alignmentList = runConllParser(inputdir, extension)
		print("Done for @ConllParser, time passed %.2fs" % (time.time() - timer))
	else:
		raise argparse.ArgumentTypeError("Incorect mode, must be stanford|conll")
	return treeList, alignmentList

def strAsThreshold(string):
#	print(string)
	items = re.split("[ ,\.\-_]", string.strip("][ "))
#	print(items)
	return [int(val) for val in items]

if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Create training examples from resource data.')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input files')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-t','--tagdir', type=str, default="all_tag.txt", help='location of the tag file containing both POStag and dependency, default all_tag.txt')
	parser.add_argument('-m','--mode', type=str, default="conll", help='parser file mode (stanford|conll), default conll')
	parser.add_argument('-x','--output_extension', type=str, default="txt", help='output file mode (txt|arff|rule|model|dep), default txt')
	parser.add_argument('-e','--embedding_word', type=str, help='embedding word file')
	parser.add_argument('--parser_extension', type=str, default="parser", help='file extension for parser, default parser')
	parser.add_argument('--default_word', type=str, default="*UNKNOWN*", help='key in embedding_word standing for unknown word. Only in arff mode')
	parser.add_argument('--full_load', action='store_true', help='get all vector for words in embedding_word file. Not recommended for large word file. Only in arff mode')
	parser.add_argument('--min_cases', type=int, default=-1, help='number of cases needed to be recorded, default -1(all), only in rule mode')
	parser.add_argument('--string_mode', action='store_true', help='try to convert to string beforehand, only usable in txt/rule mode to reduce memory usage')
	parser.add_argument('--split_rule', action='store_true', help='split the rule file into PAC/SIB file, rule mode only')
	parser.add_argument('--split_by_punctuation', action='store_true', help='split the children in groups in punctuation, model mode only')
	parser.add_argument('--percentage_stat', action='store_true', help='stat each recorded possibility, model mode only')
	parser.add_argument('--model_size', type=int, default=1, help='the amount of models to be recorded, each correspond to the amount of children')
	parser.add_argument('--model_rule_percentage', type=float, default=None, help='If specified, model(n) files will only display the highest non-duplicate swap rules above this percentage.')
	parser.add_argument('--model_rule_threshold', type=strAsThreshold, default=None, help='If specified, only the cases with value above the specified threshold are allowed into the model file. Only for output mode model')
	parser.add_argument('--debug', action='store_true', help='print debug information to both terminal and file, currently dep only')
	parser.add_argument('--injection_dep', action='store_true', help='use injection scheme for dep instead of greedy')
	parser.add_argument('--bracket_size', type=int, default=5, help='the minimum brackets for bracketPreferedSize, 1 to do best, default 5, dep onlt')
	args = parser.parse_args()
	# args.tagdir = "all_tag.txt"
	# args.embedding_word = "data\syntacticEmbeddings\skipdep_embeddings.txt"
	# args.default_word = "*UNKNOWN*"
	if(args.outputdir is None):
		args.outputdir = args.inputdir
	
	if(args.model_rule_threshold is not None):
		assert len(args.model_rule_threshold) == args.model_size, "Mismatched between rule_threshold and size: {} and {}".format(args.model_rule_threshold, args.model_size)

	timer = time.time()
	if(args.output_extension != 'model'):
		treeList, alignmentList = createIterableByMode(args.mode, args.inputdir, args.parser_extension, timer)
	
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
	elif(args.output_extension == 'model'):
		# Handle the creation of iterables by self
		for i in range(1, args.model_size + 1):
			treeList, alignmentList = createIterableByMode(args.mode, args.inputdir, args.parser_extension, timer)
			allRelation = createRelationDataByModelType(i, treeList, alignmentList, splitByPunctuation=args.split_by_punctuation)
			# allRelation = dict((k, v) for k, v in allRelation.items() if v[1] >= args.min_cases)
			print("Done for @createRelationDataByModelType(model child=%d), time passed %.2fs" % (i, time.time() - timer))
			writeAllModelRelationToFile(i, args.outputdir, allRelation, writePercentage=args.percentage_stat, writeModelPercentageValue=args.model_rule_percentage, writeModelThreshold=args.model_rule_threshold)
			del allRelation
			print("Done for @writeAllModelRelationToFile(model child=%d), time passed %.2fs" % (i, time.time() - timer))
	elif(args.output_extension == 'dep'):
		allRelation = createDepRelationByModelType(treeList, alignmentList, bracketPreferedSize=args.bracket_size, debug=args.debug, injectionMode=args.injection_dep)
		print("Done for @createDepRelationByModelType, time passed %.2fs" % (time.time() - timer))
		writeDepRelationToFile(allRelation, args.outputdir, debug=args.debug)
		print("Done for @writeDepRelationToFile, time passed %.2fs" % (time.time() - timer))
	else:
		raise argparse.ArgumentTypeError("Incorect output_extension, must be txt|arff|rule|model")
	# Execute script
