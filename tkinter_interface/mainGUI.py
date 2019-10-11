# from tkinter import Tk, Label, Button, Frame, Text, Scrollbar
import sys, time, os 
import tkinter as tki
import tkinter.messagebox as MessageBox
import tkinter.filedialog as FileDialog
import converter_interface 
#import rawPhrase as raw_converter
from trie import TrieNode as Node

counter = 0
dir_path = os.path.dirname(os.path.realpath(__file__))

DEFAULT_BINARY_FILE = "\\combinedDict.tmp"
DEFAULT_CONVERT_SOURCE = "\\1728-2.txt"
MAX_PHRLEN = 8

colorRotation = ["green","blue","red","black","white"]
COLOR_NUM = 5

DEFAULT_WIDTH = 1650
WINDOW_RATIO = 0.8
SAVE_BINARY_WHEN_CLOSING = False
SEPARATOR =('', ' ')
def getFittingTextSize():
	width = WINDOW_RATIO

class MainGUI:
	def __init__(self, master):
		self.master = master
		master.title("Phrase Translator 2.0")
		self.prefixTree = None
		self.fullDict = None
		self.combinedDict = None
		
		self.sourceFrame = HighlightableText(master, self.master, mode='grid', width=450, height=450, row=0, column=0, spacingType='', side=tki.LEFT)
		master.columnconfigure(0, weight=8)
		self.translatedFrame = HighlightableText(master, self.master, mode='grid', width=1200, height=450, row=0, column=1, columnspan=2, spacingType=' ', side=tki.RIGHT)
		master.columnconfigure(1, weight=3)
		master.rowconfigure(0, weight=1)
		
		#self.loadSavedTree("tree.tmp")
		self.controller = ControlPane(master, row=1, column=0, cfCom=self.translateFromSource, ctCom=self.clearOutCurrentTree,
			stCom=self.saveCurrentTree, ltCom=self.loadSavedTree, rdCom=self.readDictionary, toggleEdit=self.toggleEdit, aciCom=self.getIdxAndData,
			atCom=self.addPhraseToTree, clCom=self.clearAllText, scvCom=self.saveConversion)
			
		self.log = LogPane(master, row=1, column=2, mode='grid', width=50, height=15)
		
		self.log.callChangeLogDisplay("LOG:", LogPane.MODE_REP)
		# allWidth = master.winfo_width()
		allWidth = 165
		
		self.sourceFrame.txt.config(width=allWidth * 3 // 11)
		self.translatedFrame.txt.config(width=allWidth * 8 // 11)
		
		self.converter = converter_interface.ConverterObject("raw")
		self.converter.bindLoggingFunction(self.log.callChangeLogDisplay)
		if(SAVE_BINARY_WHEN_CLOSING):
			master.protocol("WM_DELETE_WINDOW", self.handleClosing)
	
	def handleClosing(self):
		self.saveCurrentTree(self.controller.treeFileDir.get())
		self.master.destroy()
	
	def clearAllText(self, disableAfter=False):
		self.sourceFrame.clearAllText()
		self.translatedFrame.clearAllText()
		if(disableAfter):
			self.sourceFrame.disableEditingText()
			self.translatedFrame.disableEditingText()
	
	def populateFrame(self, dataTupleOrList, fullDisplay=False):
		if(isinstance(dataTupleOrList, tuple)):
			convData, rawData = dataTupleOrList
		else:
			sourceSeparator = SEPARATOR[0]
			convData = []
			rawData = ''
			for raw, conv, _ in dataTupleOrList:
				convData.append(conv)
				rawData += raw + sourceSeparator
		
		self.sourceFrame.enableEditingText()
		self.translatedFrame.enableEditingText()
		for phr in rawData:
			self.sourceFrame.addTextToFrame(phr)
		for phr in convData:
			if(isinstance(phr, str)):
				self.translatedFrame.addTextToFrame(phr)
			elif(fullDisplay):
				self.translatedFrame.addTextToFrame("[{}]".format(phr))
			else:
				self.translatedFrame.addTextToFrame(phr.split("/")[0])
		self.sourceFrame.disableEditingText()
		self.translatedFrame.disableEditingText()
	
	def loadSavedTree(self, fileDir):
		timer = time.time()
		# self.prefixTree = Node(None, None, None).unpackDataToNode(fileDir)
		self.converter.loadBinaryData(fileDir)
		self.converter.writeToLog("Load completed, time passed %s" % (time.time() - timer))
		
	def saveCurrentTree(self, fileDir):
		timer = time.time()
		# self.prefixTree.packDataToFile(fileDir)
		self.converter.saveBinaryData(fileDir)
		self.converter.writeToLog("Save completed, time passed %s" % (time.time() - timer))
		
	def translateFromSource(self, sourceDir, clipboardSource=False):
		timer = time.time()
		
		if(clipboardSource or sourceDir is None):
			source = self.master.clipboard_get()
			if(not isinstance(source, str)):
				source = ""
		else:
			source = self.converter.openFile(sourceDir, 'r')
		self.converter.writeToLog("Source loaded, string size: %d, time passed %s" % (len(source), time.time() - timer))
		output = self.converter.convert(source, ruleSet={ converter_interface.numberRuleProper, converter_interface.numberRuleBackup }, separator=SEPARATOR)
		convData = []
		rawData = []
		for raw, conv, _ in output:
			convData.append(conv)
			rawData.append(raw)
		
		self.clearAllText()
		self.currentData = (convData, rawData)
		# self.converter.writeToLog("Translate done, time passed %s" % (time.time() - timer))
		self.populateFrame(self.currentData)
		self.converter.writeToLog("Translate and write completed, time passed %s" % (time.time() - timer))
	
	def saveConversion(self, targetDir, tag='default'):
		if(isinstance(targetDir, str)):
			try:
				targetDir = open(targetDir, "w", encoding="utf-8")
			except IOError:
				self.converter.writeToLog("File path {} cannot be opened to write".format(targetDir))
				return
		convertArray, _ = self.currentData
		
		for phr in convertArray:
			if(isinstance(phr, converter_interface.TranslateData) and self.converter.mode == "raw"):
				targetDir.write(phr.getValue(tag))
			elif(isinstance(phr, str)):
				targetDir.write(phr.split("/")[0] + ' ')
		targetDir.close()
	
	def readDictionary(self, dictDir, readMode=True):
		if(readMode):
			self.converter.loadTxtData(dictDir)
		else:
			self.converter.saveTxtData(dictDir)
		
	def toggleEdit(self):
		# Todo check later
		return
	
	def getIdxAndData(self):
		idx = self.translatedFrame.txt.index(tki.INSERT)
		if(idx is None):
			position = -1
		else:
			position = self.converter.binarySearch(self.translatedFrame.sortedSearchArray, idx, self.compareIndexString)
		convData, rawData = self.currentData
		rawSpacing = self.sourceFrame.spacing
		self.converter.writeToLog("Current spotted idx: {}, position {}".format(idx, position))
		return position, convData, rawData, rawSpacing
		
	def clearOutCurrentTree(self):
		# del self.prefixTree
		self.converter.clearData()
	
	def addPhraseToTree(self, source, output, isName=False, tag='default', useDataClass=True):
		# self.converter.addPhraseToTree(self.prefixTree, (source,output), True)
		if("_delete_" in output):
			tag = "delete"
		elif("_replace_" in output):
			tag = replace
			output = output.replace("_replace_", "").strip()
		updateTuple = (source, output)
		updateDictIdx = 0 if not isName else 1
		self.converter.modifyData(updateTuple, tag, updateDict=updateDictIdx)
	
	def compareIndexString(self, arg1, arg2):
		arg1 = arg1.split('.')
		arg2 = arg2.split('.')
		if(int(int(arg1[0])) > int(arg2[0])):
			return True
		elif(int(int(arg1[0])) < int(arg2[0])):
			return False
		else:
			return (int(arg1[1])) > int(arg2[1])

class HighlightableText:
	def __init__(self, parent, root=None, **kwargs):
		if(root is None):
			root = parent
		side = kwargs.get('side', tki.LEFT)
		width = kwargs.get('width', 500)
		height = kwargs.get('height', 450)
		row = kwargs.get('row', 0)
		column = kwargs.get('column', 0)
		columnspan = kwargs.get('columnspan', 1)
		mode = kwargs.get('mode', 'pack')
		self.spacing = kwargs.get('spacingType', ' ')
		self.root = root
		
		overallFrame = tki.Frame(parent)
		if(mode == 'grid'):
			overallFrame.grid(row=row, column=column, columnspan=columnspan, sticky="nsew")
			overallFrame.grid_propagate(False)
		else:
			overallFrame.config(width=width, height=height)
			overallFrame.pack(side=side, fill=tki.X)
			overallFrame.pack_propagate(False)
		
		self.outerFrame = tki.Frame(overallFrame, width=width, height=height)
		# self.outerFrame.grid(row=row, column=column)
		self.outerFrame.pack(side=side, expand=True, fill=tki.Y)
		self.outerFrame.pack_propagate(True)
		
		#self.fullText = tki.StringVar()
		self.txt = tki.Text(self.outerFrame)
		#self.txt.config(wrap='word')
		#if(mode == 'grid'):
		#	self.txt.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
		#else:
		self.txt.pack(expand=1, side=side, fill=tki.BOTH)
		#self.txt.place(x=0, y=0)
		self.sortedSearchArray = []
		
		side = tki.LEFT if(side == tki.RIGHT) else tki.RIGHT
		scrollb = tki.Scrollbar(overallFrame, command=self.txt.yview, orient=tki.VERTICAL)
		#scrollb.grid(row=row, column=column+1, sticky="nsew")
		scrollb.pack(expand=0, side=side, fill=tki.Y) # (fill=tki.BOTH, side=(tki.LEFT if side==tki.RIGHT else tki.RIGHT))
		# self.txt['yscrollcommand'] = scrollb.set
		self.txt.config(yscrollcommand=scrollb.set)
	
	def addTextToFrame(self, text, callback=None):
		global counter
		counter += 1
		"""if('\n' in text and len(text) > 1):
			text = text.replace('\n','')
		else:
			text = '\n\n'"""
		currentEnd = self.txt.index("end-1c")
		# self.converter.writeToLog("tag {} - bg {} & fg {}".format(currentEnd, colorRotation[counter % 4], colorRotation[(counter+1) % 4]))
		self.txt.insert(tki.END, text + self.spacing)
		# self.txt.tag_add(len(self.sortedSearchArray),currentEnd)
		# self.txt.tag_config(len(self.sortedSearchArray), background=colorRotation[counter % 4], foreground=colorRotation[(counter+1) % 4])
		self.sortedSearchArray.append(currentEnd)
	
	def disableEditingText(self):
		# Temporary disable writing
		self.txt.config(state=tki.DISABLED)
	
	def enableEditingText(self):
		self.txt.config(state=tki.NORMAL)
	
	def clearAllText(self):
		self.txt.config(state=tki.NORMAL)
		self.txt.delete('1.0', tki.END)
		self.sortedSearchArray= []
	
class ControlPane:
	def __init__(self, root, **kwargs):
		self.root = root
		row = kwargs.get('row', 0)
		column = kwargs.get('column', 0)
		
		self.mainFrame = tki.Frame(root, relief="sunken")
		self.mainFrame.grid(row=row,column=column)
		# self.mainFrame.pack_propagate(False)
		
		self.toggleEdit = kwargs.get('eCom', None)
		self.editButton = tki.Button(self.mainFrame, text="Edit", command=self.executeToggleEdit)
		self.editButton.grid(row=0, column=0, sticky='n')
		
		self.clearTree = kwargs.get('ctCom', None)
		self.clearTreeButton = tki.Button(self.mainFrame, text="Clear Tree", command=self.clearTreeCommand)
		self.clearTreeButton.grid(row=0, column=1, sticky='n')
		
		self.clearFrame = kwargs.get('clCom', None)
		self.clearTextButton = tki.Button(self.mainFrame, text="Clear Text", command=self.clearFrame)
		self.clearTextButton.grid(row=0, column=2, sticky='n')
		
		self.clearTextButton = tki.Button(self.mainFrame, text="Convert from Clipboard", command=self.executeConvertFromClipboardCommand)
		self.clearTextButton.grid(row=0, column=3, sticky='e')
		
		self.convertFileCommand = kwargs.get('cfCom', None)
		self.convertButton = tki.Button(self.mainFrame, text="Convert", command=self.executeConvertFileCommand)
		self.convertButton.grid(row=1, column=1, columnspan=2, sticky='w')
		self.convertDir = tki.Entry(self.mainFrame)
		self.convertDir.grid(row=1, column=3, sticky='ew')
		def getConvertFileDir():
			default_location = self._getDirectory(self.convertDir)
			path = self.openFileCommand("File to Convert", False, location=default_location)
			if(path == ''):
				return
			self.convertDir.delete(0, tki.END)
			self.convertDir.insert(0, path)
		pathConvertButton = tki.Button(self.mainFrame, text="Select File", command=getConvertFileDir)
		pathConvertButton.grid(row=1, column=0, sticky='w')
		
		self.readDictFileCommand = kwargs.get('rdCom', None)
		self.readDictButton = tki.Button(self.mainFrame, text="Read Dictionary", command=self.executeReadDictFileCommand)
		self.readDictButton.grid(row=2, column=1, sticky='w')
		self.dictDir = tki.Entry(self.mainFrame)
		self.dictDir.grid(row=2, column=3, sticky='ew')
		def getDictionaryFileDir():
			path = self.openFileCommand("Text dictionary file", False)
			if(path == ''):
				return
			self.dictDir.delete(0, tki.END)
			self.dictDir.insert(0, path)
		pathDictButton = tki.Button(self.mainFrame, text="Select File", command=getDictionaryFileDir)
		pathDictButton.grid(row=2, column=0, sticky='w')
		self.dictReadOrWriteMode = tki.BooleanVar()
		self.dictReadOrWriteMode.set(True)
		checkBox = tki.Checkbutton(self.mainFrame, text='ReadMode?', variable=self.dictReadOrWriteMode)
		checkBox.grid(row=2, column=2, sticky='es')
		
		self.saveTreeCommand = kwargs.get('stCom', None)
		self.saveTreeButton = tki.Button(self.mainFrame, text="Save to Binary", command=self.executeSaveTreeFileCommand)
		self.saveTreeButton.grid(row=3, column=1, sticky='w')
		self.loadTreeCommand = kwargs.get('ltCom', None)
		self.loadTreeButton = tki.Button(self.mainFrame, text="Load from Binary", command=self.executeLoadTreeFileCommand)
		self.loadTreeButton.grid(row=3, column=2, sticky='w')
		self.treeFileDir = tki.Entry(self.mainFrame)
		self.treeFileDir.grid(row=3, column=3, sticky='ew')
		def getBinaryFileDir():
			path = self.openFileCommand("Binary file", True)
			if(path == ''):
				return
			self.treeFileDir.delete(0, tki.END)
			self.treeFileDir.insert(0, path)
		pathDictButton = tki.Button(self.mainFrame, text="Select File", command=getBinaryFileDir)
		pathDictButton.grid(row=3, column=0, sticky='w')
		
		self.convertDir.insert(0, dir_path)
		self.dictDir.insert(0, dir_path)
		
		if(os.path.isfile(dir_path + DEFAULT_BINARY_FILE)):
			self.treeFileDir.insert(0, dir_path + DEFAULT_BINARY_FILE)
		else:
			self.treeFileDir.insert(0, dir_path)
		
		self.viewFrame = tki.Frame(root, relief="sunken")
		self.viewFrame.grid(row=row,column=column+1)
		
		self.listChoiceBox = []
		self.listCurrentSelected = []
		#for i in range(0,10):
		#	self.listCurrentSelected.append(tki.StringVar())
		#	self.listChoiceBox.append(tki.OptionMenu(self.viewFrame, self.listCurrentSelected[i], []))
		#	self.listChoiceBox[i].grid(row=1, column=i+1, sticky='n')
		self.currentChoiceContext = tki.Text(self.viewFrame, width=50, height=4)
		self.currentChoiceContext.grid(row=0, column=1, rowspan=1, columnspan=9)
		self.translatedContext = tki.Text(self.viewFrame, width=50, height=4)
		self.translatedContext.grid(row=1, column=1, rowspan=1, columnspan=9)
		
		self.askCurrentIndexCommand = kwargs.get('aciCom', None)
		self.showClusterButton = tki.Button(self.viewFrame, text="Show", command=self.showCluster)
		self.showClusterButton.grid(row=4, column=1, columnspan = 5)
		
		self.submitChangeCommand = kwargs.get('scCom', None)
		self.submitChangeButton = tki.Button(self.viewFrame, text="Done", command=self.submitChangeToCluster)
		self.submitChangeButton.grid(row=4, column=6, columnspan = 6)
		
		self.addToTreeCommand = kwargs.get('atCom', None)
		self.sourceEntry = tki.Entry(self.viewFrame)
		self.sourceEntry.grid(row=5, column=1, columnspan = 3, sticky='ws')
		self.outputEntry = tki.Entry(self.viewFrame)
		self.outputEntry.grid(row=5, column=4, columnspan = 3, sticky='s')
		self.addToTreeButton = tki.Button(self.viewFrame, text="Submit Change", command=self.executeAddToTreeCommand)
		self.addToTreeButton.grid(row=5, column=7, columnspan = 2, sticky='es')
		self.isName = tki.BooleanVar()
		checkBox = tki.Checkbutton(self.viewFrame, text='Is Name?', variable=self.isName)
		checkBox.grid(row=5, column=9, sticky='es')
		
		self.currentIndex = -1
	
	def executeToggleEdit(self):
		return self.toggleEdit()
		
	def clearTreeCommand(self):
		if(MessageBox.askokcancel("Clear Built Tree", "Are you sure you want to clear out the read data? \
You will need to import new data before trying to convert again.", parent=self.root, icon=MessageBox.WARNING)):
			return self.clearTree()
		
	def openFileCommand(self, titleName, useBinary=False, location=dir_path):
		permitType = [("Text files (.txt)", "*.txt"),("Binary save files (.tmp)", "*.tmp"),("All files", "*.*")]
		if(useBinary):
			files = (permitType[1], permitType[2])
		else:
			files = (permitType[0], permitType[2])
		return FileDialog.askopenfilename(initialdir=location, title=titleName, filetypes=files)
	
	def executeSaveFileCommand(self):
		savePath = FileDialog.asksaveasfilename(initialdir = dir_path,title = "Save",filetypes = (("Text files (.txt)","*.txt")))
		self.saveFileCommand(savePath)
	
	def executeConvertFileCommand(self):
		targetDir = self.convertDir.get()
		# quality-of-life: the convertDir location will be update at its 
		return self.convertFileCommand(targetDir)
	
	def executeConvertFromClipboardCommand(self):
		return self.convertFileCommand(None, True)
	
	def executeReadDictFileCommand(self):
		return self.readDictFileCommand(self.dictDir.get(), self.dictReadOrWriteMode.get())
	
	def executeSaveTreeFileCommand(self):
		return self.saveTreeCommand(self.treeFileDir.get())
	
	def executeLoadTreeFileCommand(self):
		return self.loadTreeCommand(self.treeFileDir.get())
	
	def executeAddToTreeCommand(self):
		# self.converter.writeToLog("Selector isName: {}".format(self.isName.get()))
		return self.addToTreeCommand(self.sourceEntry.get(), self.outputEntry.get(), self.isName.get())
		
	def submitChangeToCluster(self):
		return self.submitChangeCommand()
		
	def showCluster(self):
		position, convData, rawData, rawSpacing = self.askCurrentIndexCommand()
		if(position == -1):
			self.converter.writeToLog("Position not found @askCurrentIndexCommand function")
			return
		global counter
		counter += 1
		# Show 4 words before and 5 words after position
		if(position < 0):
			return
		if(position < 5):
			position += 4
		elif(position > len(rawData)-5):
			position = len(rawData)-5
		
		self.currentChoiceContext.delete('1.0', tki.END)
		self.translatedContext.delete('1.0', tki.END)
		for i in (range(0, 10)):
			# 0 collerate to position-4
			dataPos = position-4+i
			# populate choiceBox
			# self.listChoiceBox[i]['menu'].delete(0, tki.END)
			#val = i
			#for st in convData[dataPos].split('/'):
			#	self.listChoiceBox[val]['menu'].add_command(label=st, command=lambda choice=st:self.listCurrentSelected[val].set(choice))
			#self.listCurrentSelected[val].set(convData[dataPos].split('/')[0])
			#self.listChoiceBox[val].config(bg=colorRotation[counter % 4], fg=colorRotation[(counter+1) % 4])
			currentPosition = self.currentChoiceContext.index("end-1c")
			# populate the old context file
			self.currentChoiceContext.insert(tki.END, rawData[dataPos] + rawSpacing)
			self.currentChoiceContext.tag_add(dataPos,currentPosition, tki.END)
			self.currentChoiceContext.tag_config(dataPos, background=colorRotation[counter % COLOR_NUM], foreground=colorRotation[(counter+1) % COLOR_NUM])
			
			translatedPosition = self.translatedContext.index("end-1c")
			self.translatedContext.insert(tki.END, convData[dataPos] + ' ')
			self.translatedContext.tag_add(dataPos,translatedPosition, tki.END)
			self.translatedContext.tag_config(dataPos, background=colorRotation[counter % COLOR_NUM], foreground=colorRotation[(counter+1) % COLOR_NUM])
			counter += 1
	
	def _getDirectory(self, tkItem):
		location = tkItem.get().strip()
		if(os.path.isdir(location)):
			# open at the location
			return location
		elif(os.path.isfile(location)):
			# open at the parent dir
			return os.path.dirname(location)
		else:
			return dir_path
	

class LogPane:
	MODE_ADD = "add"
	MODE_REP = "replace"
	
	def __init__(self, root, **kwargs):
		self.root = root
		row = kwargs.get('row', 0)
		column = kwargs.get('column', 0)
		width = kwargs.get('width', 20)
		height = kwargs.get('height', 5)
		side = kwargs.get('side', tki.RIGHT)
		frameDeployMode = kwargs.get('mode', 'pack')
		
		parent = tki.Frame(self.root)
		if(frameDeployMode == 'pack'):
			parent.pack(side=side)
		else:
			parent.grid(row=row, column=column)
		
		outerFrame = tki.Frame(parent, relief="sunken")
		outerFrame.pack(side=side)
		
		self.logText = tki.Text(outerFrame, width=width, height=height)
		self.logText.config(wrap='word')
		self.logText.pack(expand=True, fill=tki.BOTH)
		
		if(side == tki.RIGHT):
			side = tki.LEFT
		else:
			side = tki.RIGHT
		scrollBar = tki.Scrollbar(parent, command=self.logText.yview, orient=tki.VERTICAL)
		scrollBar.pack(fill=tki.Y, side=side)
		self.logText.config(yscrollcommand=scrollBar.set)
		self.logText.config(state=tki.DISABLED)
	
	def callChangeLogDisplay(self, text, mode="add"):
		self.logText.config(state=tki.NORMAL)
		if(mode == "replace"):
			self.logText.delete('1.0', tki.END)
		else:
			self.logText.insert(tki.END, '\n')
		self.logText.insert(tki.END, text)
		self.logText.config(state=tki.DISABLED)
		self.logText.see(tki.END)


if __name__ == "__main__":
	root = tki.Tk()
	my_gui = MainGUI(root)
	root.mainloop()
