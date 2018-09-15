import tensorflow as tf
import numpy as np

import ffBuilder as builder
import sys, io, re, random
import calculatebleu
from subprocess import call as external_script_call

from embedder import checkCapitalize, capitalize_token
def returnCapitalize(word):
	word[0] = word[0].upper()
	return word

def transformSentenceToIds(sentence, wordToIdDict, unknownWordIds, params={}, separator=" "):
	# Transform a sentences to an array of ids
	# First, strip and split by separators
	sentence = re.split(separator, sentence.strip())
	# If want to do anything, put it in params and reflect it here
	if(params.get("decapitalize", False)):
		# if decapitalize, subtitute words with <cap> + word whenever available
		checker = checkCapitalize
		subtitute = lambda word: [capitalize_token, word.lower()] if(checkCapitalize(word)) else [word]
		sentence = [token for word in sentence for token in subtitute(token)]
	elif(params.get("lowercase", False)):
		sentence = [word.lower() for word in sentence]

	# perform lookup
	sentence = [wordToIdDict.get(word) for word in sentence]
	return sentence

def transformIdsToSentence(listIds, idToWordDict, params={}, separator=" "):
	# Transform array of ids to sentences
	sentence = [idToWordDict(idx) for idx in listIds]
	# Unlike its sibling function, doesn't requre unknown token, and separator must be string instead of possible regex above
	if(params.get("decapitalize", False)):
		# If found a cap token, capitalize the word behind it and remove the cap token
		new_sentence = []
		capitalize = False
		for token in sentence:
			if(token == capitalize_token):
				capitalize = True
			elif(capitalize == True):
				new_sentence.append(returnCapitalize(token))
				capitalize = False
			else:
				new_sentence.append(token)
	elif(params.get("lowercase", False)):
		# Do nothing here
		pass
	
	return sentence

def getBleuScore(correct, value, trimComparingIfList=True):
	for correctSentence, comparingSentence in zip(correct, value):
		if(type(correctSentence) != type(comparingSentence)):
			raise ValueError("Trying to compare different types, recheck your input. Types: {} vs {}".format(type(correctSentence), type(comparingSentence)))
		
		if(not isinstance(correctSentence, str)):
			if(not isinstance(correctSentence, list)):
				raise ValueError("Type invalid @getBleuScore, must be either list or str")
			else:
				trimComparing = min(len(correctSentence), len(comparingSentence)) if(trimComparingIfList) else len(comparingSentence)
				correctSentence = " ".join(correctSentence)
				comparingSentence = " ".join(comparingSentence[:trimComparing])
		
		return calculatebleu.BLEU(comparingSentence, correctSentence)

def getWordSetFromFile(textFile):
	# brutish one-liner that work if you don't care about occurence
	unparsedWordList = textFile.read().split()
	return set(unparsedWordList)

def getWordCounterFromFile(textFile):
	unparsedWordList = textFile.read().split()
	wordDict = {}
	for item in unparsedWordList:
		wordDict[item] = wordDict.get(item, 0)
	return wordDict

def runExternalScript(scriptDir, args):
	# run the script located in scriptDir by the specified argument
	result = external_script_call([scriptDir, args])
	# the result is a CompletedSubprocess instance, streamline it
	return (result.stdout, result.stderr)

class DecisionNode:
	def __init__(self, *args, **kwargs):
		pass
		#return self.updateArgs(*args, **kwargs)
		
	def updateArgs(self, decisionFunction, actions, nextNodes, endNode=False):
		# the decision function take in the args and return the choice relating to the actions
		self._flow_function = decisionFunction
		self._list_actions = list(zip(actions, nextNodes))
		# if boolean, stop at this node. if callable, must be something that return True/False
		self._is_end = endNode
	
	def runNodeAction(self, args):
		# feed the args to the decisionFunction, and execute the concerning action
		decision = self._flow_function(args)
		try:
			executeAction, nextNode = self._list_actions[decision]
		except IndexError as e:
			print("Error trace @runNode")
			raise e
		# execute the action
		executeAction(args)
		# go to the next node
		return nextNode
	
	def isEndNode(self, args):
		isEnd = self._is_end if not callable(self._is_end) else self._is_end(args)
		if(not isinstance(isEnd, bool)):
			raise ValueError("Violated constraint @DecisionNode: return value of _is_end not Boolean: {}".format(isEnd))
		return isEnd

def mainDecisionHandler(args, processTree, recordRun=False):
	# run the process based on the specified tree
	startNode = processTree
	if(not isinstance(startNode, DecisionNode)):
		raise ValueError("processTree not pointing to a DecisionNode object")
	
	runNode = startNode
	if(recordRun):
		recordRun = []
	while(not runNode.isEndNode(args)):
		if(recordRun):
			recordRun.append(runNode)
		runNode = runNode.runNodeAction(args)
	# endNode should execute its function as well
	if(recordRun):
		recordRun.append(runNode)
	endNode = runNode.runNodeAction(args)

	return recordRun


def testRunDecision():
	firstNode = DecisionNode()
	secondNode = DecisionNode()
	thirdNode = DecisionNode()
	fourthNode = DecisionNode()
	args = {}
	# first node create a random number and add it
	default_choice = lambda args: 0
	def addRandomToArgs(args):
		args["value"] = random.randint(0, 20)
	# first node go to second node
	firstNode.updateArgs(default_choice, [addRandomToArgs], [secondNode])
	# second node see if the random number is larger than 10
	larger_than_ten = lambda args: args["value"] > 10
	# if larger, let 2 power it, else 3 power it
	def powerTo2(args):
		args["value"] = np.power(2, args["value"])
	def powerTo3(args):
		args["value"] = np.power(3, args["value"])
	# second node go to third on larger, and fourth on smaller
	secondNode.updateArgs(larger_than_ten, [powerTo2, powerTo3], [thirdNode, fourthNode])
	# third and fourth node output it name and exit
	def printThird(args):
		print("@Third node, value {}".format(args["value"]))
	def printFourth(args):
		print("@Fourth node, value {}".format(args["value"]))
	# Fourth should exit with a good checker
	fourth_checker = lambda args: args["value"] % 3 == 0
	thirdNode.updateArgs(default_choice, [printThird], [None], endNode=True)
	fourthNode.updateArgs(default_choice, [printFourth], [None], endNode=fourth_checker)
	
	# execute the test
	mainDecisionHandler(args, firstNode)

testRunDecision()
