import rawPhrase as raw_converter
from rawPhrase import TranslateData, numberRuleProper, numberRuleBackup

DEFAULT_NAME_DICT = "name.txt"
DEFAULT_PHRASE_DICT = "phrase.txt"

def _distinguishDict(loaded_data, logFunc, fileDir):
	# a wrapper on raw_converter loadDict
	if(isinstance(loaded_data, tuple) and len(loaded_data) == 2):
		logFunc("Loaded combine dict @{:s}".format(fileDir))
		return loaded_data
	elif(isinstance(loaded_data, dict)):
		logFunc("Loaded single dict, consider it as a phraseDict @{:s}".format(fileDir))
		return (loaded_data, 0)
	else:
		raise ValueError("Loaded data {} cannot be distinguised".format(loaded_data))

class ConverterObject:
	# generally handle the interface between the mainGUI and the converter 
	def __init__(self, mode):
		self.mode = mode
		if(mode == "raw"):
			# wrapped raw_converter
			def wrappedRawLoadData(fileDir):
				loaded_data = raw_converter.getDictionaryFromFile(fileDir)
				checked_data = _distinguishDict(loaded_data, self.writeToLog, fileDir)
				return checked_data
			self.loadDataFunc = wrappedRawLoadData
			self.loadTextDataFunc = raw_converter.importPairFromDefaultFile
			self.saveDataFunc = raw_converter.packDictionaryToFile
			self.saveTextDataFunc = raw_converter.writeToTxtFileFromDict
			self.convertFunc = raw_converter.convertByPhraseWithHandRule
			self.updateDataFunc = raw_converter.modifyDict
			# add option here so the functions inside raw_converter can access the writeToLog
			raw_converter.bindLoggingFunction(self.writeToLog)
		elif(mode == "neural"):
			raise NotImplementedError("Neural net version not implemented")
		else:
			raise ValueError("mode {:s} not compatible".format(mode))
		# those are default by raw_converter
		self.openFile = lambda filedir, mode: raw_converter.openFileByChardet(filedir, mode, logger=self.writeToLog)
		self.binarySearch = raw_converter.binarySearch

	def loadBinaryData(self, fileDir):
		self.data = self.loadDataFunc(fileDir)
		return self.data

	def loadTxtData(self, fileDir):
		assert self.mode == "raw", "loadTxtData only available to raw mode"
		if(os.path.isfile(fileDir)):
			self.data = (self.loadTextDataFunc(fileDir), {})
		elif(os.path.isdir(fileDir)):
			nameDictDir = os.path.join(fileDir, DEFAULT_NAME_DICT)
			phraseDictDir = os.path.join(fileDir, DEFAULT_PHRASE_DICT)
			self.data = (self.loadTextDataFunc(phraseDictDir), self.loadTextDataFunc(nameDictDir))
		return self.data
	
	def saveBinaryData(self, fileDir):
		self.saveDataFunc(self.data, fileDir)

	def saveTxtData(self, fileDir):
		assert self.mode == "raw", "saveTxtData only available to raw mode"
		if(os.path.isfile(fileDir)):
			self.writeToLog("Warning: point to a single file, exporting phraseDict only")
			self.saveTextDataFunc(self.data[0], fileDir)
		elif(os.path.isdir(fileDir)):
			nameDictDir = os.path.join(fileDir, DEFAULT_NAME_DICT)
			phraseDictDir = os.path.join(fileDir, DEFAULT_PHRASE_DICT)
			self.saveTextDataFunc(self.data[0], phraseDictDir)
			self.saveTextDataFunc(self.data[1], nameDictDir)

	def modifyData(self, updateTuple, updateType, updateDict=0):
		assert self.mode == "raw", "modify on-the-fly only available to raw mode"
		key, value = updateTuple
		if(updateType == "delete"):
			phraseFound = self.data[0].pop(key, None)
			nameFound = self.data[1].pop(key, None)
			self.writeToLog("Phrase {:s} deleted, in phraseDict: {}, in nameDict {}".format(key, phraseFound, nameFound))
		elif(updateType == "default"):
			self.updateDataFunc(self.data[updateDict], key, value, replace=False)
			self.writeToLog("Phrase {:s} updated with {:s}, dictIdx {:d}".format(key, value, updateDict))
		elif(updateType == "replace"):
			self.updateDataFunc(self.data[updateDict], key, value, replace=True)
			self.writeToLog("Phrase {:s} replaced with {:s}, dictIdx {:d}".format(key, value, updateDict))

	def clearData(self):
		del self.data
		self.writeToLog("Data deleted; revert to None")
		self.data = None

	def convert(self, targetData, **kwargs):
		assert self.data is not None, "ERROR: No data to convert"
		return self.convertFunc(targetData, self.data, )

	def bindLoggingFunction(self, function):
		self.writeLogFunc = function

	def writeToLog(self, *args, **kwargs):
		return self.writeLogFunc(*args, **kwargs)

## legacy, remove later
"""
numberDetect = '[%s]{2,}' % chineseNumberCharacters
numberReplace = ['$REGEXIN$〇$REGEX$0$REGEXOUT$', '$REGEXIN$一$REGEX$1$REGEXOUT$', '$REGEXIN$二$REGEX$2$REGEXOUT$', '$REGEXIN$三$REGEX$3$REGEXOUT$', '$REGEXIN$四$REGEX$4$REGEXOUT$', 
				'$REGEXIN$五$REGEX$5$REGEXOUT$', '$REGEXIN$六$REGEX$6$REGEXOUT$', '$REGEXIN$七$REGEX$7$REGEXOUT$', '$REGEXIN$八$REGEX$8$REGEXOUT$', '$REGEXIN$九$REGEX$9$REGEXOUT$', 
				'$REGEXIN$^(十|百|千|万|亿)[〇一二三四五六七八九]$REGEX$1$REGEXOUT$',# if 10/100/1000/10000 is at top of phrase followed by number, it is 1
				'$REGEXIN$^十$REGEX$10$REGEXOUT$', '$REGEXIN$^百$REGEX$100$REGEXOUT$', '$REGEXIN$^千$REGEX$1000$REGEXOUT$', '$REGEXIN$^万$REGEX$10000$REGEXOUT$', 
				'$REGEXIN$^亿$REGEX$100,000,000$REGEXOUT$',# if 10/100/1000/10000 is at top of phrase not followed by number, it is the value
				'$REGEXIN$十$$REGEX$0$REGEXOUT$', '$REGEXIN$百$$REGEX$00$REGEXOUT$', '$REGEXIN$千$$REGEX$000$REGEXOUT$', '$REGEXIN$万$$REGEX$0000$REGEXOUT$', 
				'$REGEXIN$亿$$REGEX$00tr$REGEXOUT$',# if 10/100/1000/10000/10^8 is at end of phrase, it is the 0 trailing
				'$REGEXIN$十|百|千|万|亿$REGEX$$REGEXOUT$']
numberRule = raw_converter.HandRule('before_dict_tran', numberDetect, '_'.join(numberReplace), '$NUMBER$')
"""
