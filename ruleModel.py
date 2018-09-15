from exampleBuilder import *
import os, io, re, time

MAX_MODEL = 20
overallRegex = re.compile("(([\w$;,?!]+ +)+)([\d_]+-[\d_]+)")
# The regex will match the cases in .model files

def checkModelSize(modelDir, valueToCheck=MAX_MODEL):
	for size in range(1, valueToCheck+1):
		filePath = modelDir + ".model{:d}".format(size)
		if(os.path.isfile(filePath)):
			continue
		else:
			print("Model {:d} not found @filePath {!s}, will use model upto {:d}".format(size, filePath, size-1))
			return size-1
	return valueToCheck

def readModelFromFiles(modelDir, maxModelFound):
	# Read the transpose models here
	listModels = []
	def convertKey(key):
		return "_".join(key.split())
	def convertTranspose(transposeRule):
		return tuple(transposeRule.split("-"))

	for i in range(1, maxModelFound+1):
		modelFile = io.open(modelDir + ".model{:d}".format(i), 'r', encoding='utf-8')
		model = {}
		for line in modelFile.readlines():
			match = re.match(overallRegex, line)
			if(match is not None):
				ruleValue = convertKey(match.group(1))
				changeValue = convertTranspose(match.group(3))
				transposeArray = model.get(ruleValue, [])
				transposeArray.append(changeValue)
				model[ruleValue] = transposeArray
		listModels.append(model)
		modelFile.close()
	return listModels

def createDepRuleDictFromFile(filePath, bracketSeparator='->', excludeTags=[]):
	depFile = io.open(filePath, 'r', encoding='utf-8')
	listRaws = depFile.readlines()
	# to bypass : as POSTag, use an if condition
	listRaws = [(":", "()") if(item[0] == ":") else item.split(":") for item in listRaws]
	depDict = {}
	for tag, rawPos in listRaws:
		# the depDict will have tag as key and value is another dict: each dep tag is assigned an integer denoting their relative position toward each others.
			depTagOrderDict = {}
			depDict[tag.strip()] = depTagOrderDict
			barePos = rawPos.split(bracketSeparator) if(rawPos.find(bracketSeparator) >= 0) else [rawPos]
			# barePos is splitted bracket still in string, so clear out the parentheses and index them
			barePos = [rawTagList.strip("()").split() if rawTagList.find(" ") >=0 else [rawTagList.strip("()")] for rawTagList in barePos]
			correctPos = [(i, deptag) for i, tagList in enumerate(barePos) for deptag in tagList if tag not in excludeTags]
			for i, deptag in correctPos:
				depTagOrderDict[deptag.strip()] = i

	depFile.close()
	return depDict

def customizedRecursiveRun(listFunction, node, runFunction):
	# listFunction will create a list for runFunction to iterate through
	listRunner = listFunction(node)
	for item in listRunner:
		if(item is not node):
			customizedRecursiveRun(listFunction, item, runFunction)
		else:
			runFunction(node)

def runModelOnTrees(listTrees, listModels, splitByPunctuation=False, separator='_', removeRoot=False, debug=False):
	# listTrees can be taken from runConllParser, listModels from readModelFromFiles above
	
	# Prepare a list of formatter using exampleBuilder createFormatModel
	listFormatter = [createFormatModel(modelSize, separator) for modelSize in range(1, len(listModels) + 1)]
	modelMaxSize = len(listFormatter)
#	assert len(listModels) == len(listFormatter)
	# Checker function which will add nodes and its children to writeList by the correct order infered from listModels
	# By using customizedRecursiveRun, sortNodeByModel will be used as the listFunction above
	def sortNodeByModel(node):
		childrenList = splitChildrenByPunctuation(node.children) if splitByPunctuation else [node.children]
		childrenList = list(filter(lambda l: len(l) > 0, childrenList))
		if(len(childrenList) == 0):
			return [node]
		sortedChildrenList = []
		# apply the model for each sub-child separated by splitByPunctuation
		for children in childrenList:
			# get the default item setting
			nodesList = [node] + children
			currentList = sorted(nodesList, key=lambda x: x.pos)
			currentPos = [nodesList.index(item) for item in currentList]
			# if too much children, disengage and return the currentList
			if(len(children) > modelMaxSize):
				sortedChildrenList.append(currentList)
				continue
			# reorder the children if found in listModels, else keep the order as-is
			formatString, formatFunc, _ = listFormatter[len(children)-1]
			tagKey = formatFunc(formatString, node, listChild=children)
			if(tagKey not in listModels[len(children)-1]):
				# Key not found at all, continue to next iteration and use the default item
				sortedChildrenList.append(currentList)
				continue
			# search for the correct key inside the list, output None if false
			listTransposition = listModels[(len(children))-1][tagKey]
			keyTransposition = separator.join((str(idx) for idx in currentPos))
#			print("List check", listTransposition)
			correctTransposition = next((tranTo for tranFrom, tranTo in listTransposition if tranFrom == keyTransposition), None)
			if(correctTransposition is None):
				# no transposition of this kind within that key, do the same as above and go to next iteration
				sortedChildrenList.append(currentList)
			else:
				# reorder by the correctTransposition key
				correctTranspositionIdx = (int(idx) for idx in correctTransposition.split(separator))
				correctList = [nodesList[idx] for idx in correctTranspositionIdx]
				if(debug):
					print("KEY {} from {} to {}, words from {} to {}".format(tagKey, keyTransposition, correctTransposition, [item.word for item in currentList], [item.word for item in correctList]))
				sortedChildrenList.append(correctList)
		# merge the keys by priority: items belong to the same list must be in order of their list, items in different list will compete by its true pos
		# example: two subchild 1-2-0(1) and 1-0-2-3(2) turn to 0-1-2(1) and 1-3-0-2(2) must go 1(2) - 3(2) - 0 - 1(1) - 2(1) - 2(2), with 1(1) compare 2(2) and 2(1) compare 2(2) both in favor of 2(2) due to its greater pos
		# should not cause such complicated problem since there are punctuation inbetween, but just to be sure
		fullSorted = sortedChildrenList[0]
		for otherChilds in sortedChildrenList[1:]:
			# insert node-by-node, each node must be (1) at the index larger than the one before, (2) at the correct positioning respective to the parent node, and (3) as close to the original pos to another node as possible
			isLeftNode = True
			lastInsertIdx = -1
			for item in otherChilds:
				if(item is node):
					isLeftNode = False
					continue
				parentPosition = fullSorted.index(node)
				if(isLeftNode):
					# search for the first larger pos in the left side of the list, insert before parentNode if fail
					insertPosition = next((max(i, lastInsertIdx+1) for i, childNode in enumerate(fullSorted[:parentPosition]) if childNode.pos > item.pos), parentPosition)
					fullSorted.insert(insertPosition, item)
					lastInsertIdx = insertPosition
				else:
					# do the same thing for the right side, and insert into the end of the list if search fail
					insertPosition = next((max(i, lastInsertIdx+1) for i, childNode in enumerate(fullSorted[parentPosition+1:]) if childNode.pos > item.pos), len(fullSorted))
					fullSorted.insert(insertPosition, item)
					lastInsertIdx = insertPosition
		# Once done, put any punctuation into the best position available inside the fullSorted list
		# That being said, just put them the same way as above - before meeting a larger pos once sorted
		if(splitByPunctuation):
			listPunctuation = [item for item in node.children if item.isPunctuation()]
			for item in listPunctuation:
				insertPosition = next((i for i, childNode in enumerate(fullSorted) if childNode.pos > item.pos), len(fullSorted))
				fullSorted.insert(insertPosition, item)
		return fullSorted

	listResult = []
	for tree in listTrees:
		# Constructed trees will run customizedRecursiveRun to establish the correct orders
		orderedWordList = []
#		def insertIntoList(node):
#			orderedWordList.append(node)
#			print("added node ({}-{})".format(node.pos, node.word))
		customizedRecursiveRun(sortNodeByModel, tree, lambda node: orderedWordList.append(node))
		if(removeRoot):
			orderedWordList = filter(lambda node: node.pos > 0, orderedWordList)
		if(debug):
			oldList = sorted(tree.getAllNodeInTree(), key=lambda node: node.pos)
			print("From old positioned list {!s} to {!s}: ".format([node.word for node in oldList], [node.word for node in orderedWordList]))
		listResult.append([str(node.word) for node in orderedWordList])
		deleteTree(tree)
	
	# print("Result length ", len(listResult))
	return listResult

def applyDepRuleOnTrees(listTrees, depRuleDict, splitByPunctuation=False, debug=False, removeRoot=False):
	def reorderByDep(node):
		listNodes = list(node.children) + [node]
		if(len(listNodes) <= 1):
			return listNodes
		if(splitByPunctuation):
			splittedListNodes = splitChildrenByPunctuation(listNodes, keepPunct=True)
#			def convertListNodesToListString(lNode):
#				return [node.word for node in lNode]
#			print("Old: {}, Splitted: {}".format(convertListNodesToListString(sorted(listNodes, key=lambda item: item.pos)), [convertListNodesToListString(subList) for subList in splittedListNodes]))
		else:
			splittedListNodes = [listNodes]
		depTagOrderDict = depRuleDict[node.tag.strip()]
		defaultIdx = depTagOrderDict['self'] if 'self' in depTagOrderDict else 0
		sortedListNodes = []
		for subListNodes in splittedListNodes:
			# reorder each sublist instead, and feed them into the sortedListNodes
			listNodesIdx = [depTagOrderDict.get(child.dependency, defaultIdx) if child is not node else defaultIdx for child in subListNodes]
			# Reorder base on the idx previously gotten, all the while preserving in-bracket positioning
			subListNodes = zip(listNodesIdx, subListNodes)
			subListNodes = sorted(subListNodes, key=lambda x: (x[0], x[1].pos))
			sortedListNodes.append(subListNodes)
		# flatten the sortedListNodes back to one dimensional list
		listNodes = [node[1] for subListNodes in sortedListNodes for node in subListNodes]
		
		return listNodes
	
	listResult = []
#	counter = 0
	for tree in listTrees:
#		counter += 1
#		if(counter==14):
#			sys.exit()
		orderedWordList = []
		customizedRecursiveRun(reorderByDep, tree, lambda node: orderedWordList.append(node))
		if(removeRoot):
			orderedWordList = list(filter(lambda node: node.pos > 0, orderedWordList))
		if(debug):
			oldList = sorted(tree.getAllNodeInTree(), key=lambda node: node.pos)
			print("From old positioned list {!s} to {!s}: ".format([node.word for node in oldList], [node.word for node in orderedWordList]))
			assert len(oldList) == len(orderedWordList) + (1 if removeRoot else 0)
		listResult.append([str(node.word) for node in orderedWordList])
		deleteTree(tree)
	return listResult

def printResultToFile(fileoutDir, listOutput, spacing=" ", wrapper="{!s}", removeRoot=False):
	# Warning: removeRoot option is unsafe if there is root rules. Use the removeRoot on the runModelOnTrees instead
	fileOut = io.open(fileoutDir, 'w', encoding='utf-8')
	
	for item in listOutput:
		if(removeRoot):
			item = item[1:]
		# item should be an array of words, join them over
		item = spacing.join((wrapper.format(word) for word in item))
		fileOut.write(item + "\n")
	
	fileOut.close()

def testFunction():
	raise Exception("Broken, need rewrite")
	modelDir = "CICLING/en-vi/train_75percent"
	modelSize = 5
	modelSize = checkModelSize(modelDir, modelSize)
	print("CheckModelSize with val 5: to {}, expected 4".format(modelSize))
	listModels = readModelFromFiles(modelDir, modelSize)
	print("Created list of models: ", listModels)
	inputDir = "CICLING/en-vi/test"
	extension = "clean.en.conllx"
	listTrees = runConllParser(inputDir, extension=extension, alignExtension=None)

if __name__ == '__main__':
	# testFunction()
	parser = argparse.ArgumentParser(description='Run the synthesized models on .')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input file')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-m','--mode', type=str, default='model', help='this variable specify which kind of reordering are to be done. "model" for .model(n) file, dep for .dep file')
	parser.add_argument('--modeldir', type=str, default=None, help='location of the model files. These files must have extension .model(n) (e.g. .model2) and start from 1. Will only be used with --mode model')
	parser.add_argument('--depdir', type=str, default=None, help='location of the dep file, must have extension .dep and without debug information. Will only be used with --mode dep')
	parser.add_argument('-d','--debug', action='store_true', help='print the swapping process and result to the stdout')
	parser.add_argument('--input_extension', type=str, default="conll", help='file extension for the input processed file, default conll')
	parser.add_argument('--output_extension', type=str, default="txt", help='output file extension, default txt')
	parser.add_argument('--model_size', type=int, default=MAX_MODEL, help='the amount of models to be used, will use upto 20 models found if unspecified')
	parser.add_argument('--split_by_punctuation', action='store_true', help='split the children in groups by punctuation when infering')
	parser.add_argument('--keep_root', action='store_true', help='Keep the root node in the infering process')
	args = parser.parse_args()
	# Add customized spacing and wrapper arguments for the export
	if(args.outputdir is None):
		args.outputdir = args.inputdir
	
	startTimer = time.time()
	def getTimer():
		return time.time() - startTimer
	
	if(args.mode == 'model'):
		if(args.modeldir is None):
			args.modeldir = args.inputdir
		
		modelSize = checkModelSize(args.modeldir, args.model_size)
		if(modelSize <= 0):
			raise FileNotFoundError("Error when checking args.modeldir: No model files found. Make sure the path is valid and the models files started with .model1")
		
		listModels = readModelFromFiles(args.modeldir, modelSize)
		print("All models read, model found %d, time passed %.2fs." % (len(listModels), getTimer()))
		
		listTrees = runConllParser(args.inputdir, extension=args.input_extension, alignExtension=None)
		print("List of trees generated, time passed %.2fs." % getTimer())
		
		result = runModelOnTrees(listTrees, listModels, splitByPunctuation=args.split_by_punctuation, debug=args.debug, removeRoot=not args.keep_root)
		print("Reordering done for %d sentences, time passed %.2fs." % (len(result), getTimer()))
	elif(args.mode == 'dep'):
		if(args.depdir is None):
			args.depdir = args.inputdir
		depRuleDict = createDepRuleDictFromFile(args.depdir + '.dep', bracketSeparator='->', excludeTags=['punct'])
		print("RuleDict fully read, total of {:d} tags, time passed {:.2f}s".format(len(depRuleDict), getTimer()))
		
		listTrees = runConllParser(args.inputdir, extension=args.input_extension, alignExtension=None)
		print("List of trees generated, time passed %.2fs." % getTimer())
		
		result = applyDepRuleOnTrees(listTrees, depRuleDict, splitByPunctuation=args.split_by_punctuation, debug=args.debug, removeRoot=not args.keep_root)
		print("Reordering done for %d sentences, time passed %.2fs." % (len(result), getTimer()))
	else:
		raise argparse.ArgumentError("Mode string {} not recognized.".format(args.mode))
	exportDir = args.outputdir + '.' + args.output_extension
	printResultToFile(exportDir, result) 
	print("Result exported to file %s(overwrite), time passed %.2fs." % (exportDir, getTimer()))
