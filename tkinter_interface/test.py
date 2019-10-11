import sys, getopt, re, time
from rawPhrase import *
from trie import TrieNode as Node

fullTimer = time.time()
USE_TREE = False
LOAD_TXT = False
DEBUG = False
VALID_CHECK = True
FULL_PHRASES = False

fullUppercaseRegex = re.compile("[AĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]")
filePath = "phrase.txt"
fileRedoPath = "phrase_sorted.txt"

# Open file and read
file = open(filePath, "r", encoding="utf-8")
source = file.readlines()
source = filter(lambda x: not re.match(r'^\s*$', x), source)
# 3 arrays for ERROR, then uppercase, then normal
sourceSplit = ([], [], [])
for pair in source:
	if(pair.find("ERROR") >= 0):
		sourceSplit[0].append(pair)
	elif(fullUppercaseRegex.search(pair)):
		sourceSplit[1].append(pair)
	else:
		sourceSplit[2].append(pair)
all = []
for sen in sourceSplit:
	all.extend(sen)
file.close()
# Join up on a new file
file = open(fileRedoPath, "w", encoding='utf-8')
for sentence in all:
	file.write(sentence)
file.close()

sys.exit(0)

if(USE_TREE):
	if(LOAD_TXT):
		try: 
			dictFile = open("txt/LacViet.txt", "r", encoding="utf-8")
		except IOError:
			print('Cannot read file LacViet')
			sys.exit()
		phraseDict = importPairFromDictionaryFile(dictFile, True)
		dictFile.close()
		try: 
			dictFile = open("txt/VietPhrase.txt", "r", encoding="utf-8")
		except IOError:
			print('Cannot read file VietPhrase')
			sys.exit()
		phraseDict = importPairFromDefaultFile(dictFile, phraseDict, True)
		dictFile.close()
		try: 
			dictFile = open("txt/Names.txt", "r", encoding="utf-8")
		except IOError:
			print('Cannot read file Names')
			sys.exit()
		nameDict = importPairFromDefaultFile(dictFile, True)
		dictFile.close()
		
		#prefixTree = convertToPrefixTree(fullDict)
		#prefixTree.packDataToFile("tree_manual.tmp")
		packDictionaryToFile((phraseDict, nameDict), "combinedDict.tmp")
		phraseDict.update(nameDict)
		fullDict = phraseDict
	else:
		prefixTree = Node(None, None, None)
		prefixTree = prefixTree.unpackDataToNode("tree_manual.tmp")
		print("Unpack completed, time passed %s" % (time.time() - fullTimer))
else:
	phraseDict, nameDict = getDictionaryFromFile("combinedDict.tmp")

if(DEBUG):
	timer = time.time()
	try: 
		debugFile = open("debug.txt", "w", encoding="utf-8")
	except IOError:
		print('Cannot open debug file')
		sys.exit()
	printTreeToFile(prefixTree, prefixTree, debugFile)
	debugFile.close()
	print("Reprint to file completed, time passed %s" % (time.time() - timer))
	
if(not DEBUG and VALID_CHECK):
	timer = time.time()
	try: 
		wordFile = open("txt/Viet74K.txt", "r", encoding="utf-8")
		debugFile = open("debug.txt", "w", encoding="utf-8")
	except IOError:
		print('Cannot open debug file')
		sys.exit()
		
	content = wordFile.readlines()
	content = filter(lambda x: not re.match(r'^\s*$', x), content)
	validWord = {}
	for line in content:
		validWord[line.strip().lower()] = 'null'
	debugFile.write("------INVALID-------")
	for key in phraseDict:
		keyValid = False
		allValues = None
		if(isinstance(phraseDict[key], TranslateData)):
			allValues = phraseDict[key].getAllValues()
		elif(isinstance(phraseDict[key], str)):
			if(phraseDict[key].find('/') > 0):
				allValues = phraseDict[key].split('/')
			else:
				allValues = [phraseDict[key]]
		if(allValues is not None):
			for v in allValues:
				if(v.lower() in validWord):
					keyValid = True
					validWord[v.lower()] = key
					break
			if(not keyValid):
				debugFile.write("{} = [{}]\n".format(key, "][".join(allValues)))
		elif(not keyValid):
			debugFile.write("{} = [{}]\n".format(key, phraseDict[key]))
	
	debugFile.write("------VALID-------")
	for word in validWord:
		debugFile.write("{} = {}\n".format(validWord[word], word))
	debugFile.close()
	print("Reprint to file completed, time passed %s" % (time.time() - timer))


try: 
	sourceFile = open("1728-2.txt", "r", encoding="utf-8")
except IOError:
	print('Cannot read file source')
	sys.exit()
source = sourceFile.read()
sourceFile.close()
try: 
	logFile = open("dumplog.txt", "w", encoding="utf-8")
except IOError:
	print('Cannot read file source')
	sys.exit()

#output = convertByPhrasesFromPrefixTree(source, prefixTree, logFile)
output = convertByPhraseFromDict(source, fullDict, 8, logFile)
logFile.close()

try: 
	result = open("convertedText.txt", "w", encoding="utf-8")
	for phr in output:
		if(isinstance(phr, (bytes, bytearray, str))):
			if(FULL_PHRASES):
				if(phr.find('\n') >= 0):
					result.write(phr)
				else:
					result.write("[{}]".format(phr))
			else:
				result.write(phr.split("/")[0])
				result.write(" ")
		elif(isinstance(phr, TranslateData)):
			result.write(phr.getValue())
		#else:
			# print("found abnormal value {}({})".format(phr,type(phr)))
except IOError:
	print('Cannot open file to write')
	sys.exit()

print("All process done, total cost %s" % (time.time() - fullTimer))