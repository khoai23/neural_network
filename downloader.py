import urllib2

def downloadSingleFile(saveDir, webDir, filename):
	file = open(saveDir + filename, 'w', encoding='utf-8')
	response = urllib2.urlopen(webDir)
	file.write(response.read())
	file.close()
	
def downloadFileByList(saveDir, listWebDir, listLen=-1, listFilenames=None):
	if(listLen < 0):
		listLen = len(listWebDir)
	if(listFilenames is None):
		listFilenames = [x.split()[-1] for x in listWebDir]
		for filename in listFilenames:
			if(filename.find('.') < 0):
				filename = filename + '.txt'
	for i in range(listLen):
		webDir = listWebDir[i]
		filename = listFilenames[i]
		downloadSingleFile(saveDir, webDir, filename)
	
def defaultNumberingFunc(num):
	return "{}.txt".format(num)
	
def downloadFileNumbered(saveDir, commonWebDir, length, numberingFunc=None):
	if(numberingFunc is None):
		numberingFunc = defaultNumberingFunc
	for i in range(length):
		webDir = commonWebDir + numberingFunc(i)
		filename = numberingFunc(i)
		downloadSingleFile(saveDir, webDir, filename)