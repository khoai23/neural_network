import pickle

def defaultAssignData(data):
	return data
def defaultHandleNewInput(oldData, newData):
	if(oldData is None):
		oldData = newData
	elif(isinstance(oldData, list)):
		if(isinstance(newData, list)):
			oldData.extend(newData)
		else:
			oldData.append(newData)
	elif(isinstance(oldData, int) and isinstance(newData, int)):
		oldData += newData
	return oldData

class TrieNode:
	assignData = defaultAssignData
	handleNewInput = defaultHandleNewInput
	dataNullValue = None
	
	def __init__(self, nodeKey, nodeValue, parent):
		self.nodeKey = nodeKey
		self.parent = parent
		self.data = TrieNode.assignData(nodeValue)
		self.children = []
	
	def addNode(self, newNodeOrChar):
		if(isinstance(newNodeOrChar, str)):
			newNodeOrChar = TrieNode(newNodeOrChar, TrieNode.dataNullValue, self)
		self.children.append(newNodeOrChar)
		return newNodeOrChar
	
	def buildNode(self, word, index, newVal):
		# Check as father node, create new if not available
		if(index == len(word) - 1):
			self.data = TrieNode.handleNewInput(self.data, newVal)
			return
		index += 1
		haveFittingChildNode = False
		for node in self.children:
			if(word[index] == node.nodeKey):
				node.buildNode(word, index, newVal)
				haveFittingChildNode = True
				break
		if(not haveFittingChildNode):
			newNode = self.addNode(word[index])
			newNode.buildNode(word, index, newVal)
	
	def seekNode(self, word, currIdx):
		if(index == len(word) - 1):
			return self.data
		currIdx += 1
		for node in self.children:
			if(word[index] == node.nodeKey):
				return node.seekNode(word, currIdx)
		return TrieNode.dataNullValue
		
	def seekNodeByChar(self, nextChar):
		# return node if found nextChar in children, else return None
		for node in self.children:
			if(node.nodeKey == nextChar):
				return node
		return None
	
	def getTrueWord(self, root):
	# return a tuple of ( word , listSentence )
		node = self
		word = []
		while(node is not root and node.nodeKey is not None):
			word.append(node.nodeKey)
			node = node.parent
		return (''.join(reversed(word)), self.data)
		
	def packDataToFile(self, file):
		if(isinstance(file, str)):
			file = open( file, "wb" )
		pickle.dump(self, file)
		file.close()
		
	def unpackDataToNode(self, file):
		if(isinstance(file, str)):
			file = open( file, "rb" )
		self = pickle.load(file)
		file.close()
		return self