import argparse, io

EXIT_ON_ERROR = False

def getDifferentiator(diff):
	#if(not isinstance(diff, str)):
	#	raise argparse.ArgumentTypeError("String expected")
	differentiator = diff.split('.')
	if(len(differentiator) != 2):
		raise argparse.ArgumentTypeError("Incorect differentiator, must be formatted (larger.smaller)")
	return differentiator

def printError(args):
	print(args)
	if(EXIT_ON_ERROR):
		sys.exit(0)

def parseAlignmentFromString(string):
	pairs = string.strip().split(' ')
	result = []
	for pairing in pairs:
		number = pairing.split('-')
		if(len(number) != 2):
			printError("Parsing failed: pairing {}, parsed {}".format(pairing, number))
		result.append((int(number[0]), int(number[1])))
	return result
	
def createAlignmentFromCouple(original, changed):
	original = original.split()
	changed = changed.split()
	idxRecord = list(range(len(changed)))
	result = []
	# Pop the sentence words end to end
	for idx in range(len(original)):
		word = original[idx]
		if(word in changed):
			# Found word in other, add to array
			changedIdx = changed.index(word)
			result.append((idx, idxRecord[changedIdx]))
			# Popping from the records
			changed.pop(changedIdx)
			idxRecord.pop(changedIdx)
	return result
	
def scoreFromAlignmentArray(alignment):
	if(len(alignment) == 1):
		return 1
	# Sort, to be sure
	alignment = sorted(alignment, key=lambda x: x[0]*1000+x[1])
	# Convert to single num array
	alignment = [x[1] for x in alignment]
	
	all_pair = float(len(alignment) * (len(alignment) - 1) / 2)
	con_pair = 0
	for i in range(len(alignment)-1):
		for j in range(i+1, len(alignment)):
			if(alignment[i] <= alignment[j]):
				con_pair += 1
		
	return (float(con_pair) * 2 / all_pair) - 1
	
def readTogether(inputdir, src, tgt, encoding='utf-8'):
	# Read from files
	sourceFile = io.open(inputdir + '.' + src, mode='r', encoding=encoding)
	targetFile = io.open(inputdir + '.' + tgt, mode='r', encoding=encoding)
	alignmentFile = io.open(inputdir + '.align', mode='r', encoding=encoding)
	source = sourceFile.readlines()
	target = targetFile.readlines()
	alignment = alignmentFile.readlines()
	sourceFile.close()
	targetFile.close()
	alignmentFile.close()
	
	# Bundle them together in tuple
	result = []
	if(len(source) != len(target) or len(target) != len(alignment)):
		printError("Mismatched read size, 3 file: {}-{}-{}".format(len(source), len(target), len(alignment)))
	for i in range(len(source)):
		aligment_parsed = parseAlignmentFromString(alignment[i])
		score = scoreFromAlignmentArray(aligment_parsed)
		result.append((source[i], target[i], alignment[i], score))
		
	return result
	
def createScore(inputdir, src, tgt, encoding='utf-8'):
	# Read from files
	sourceFile = io.open(inputdir + '.' + src, mode='r', encoding=encoding)
	targetFile = io.open(inputdir + '.' + tgt, mode='r', encoding=encoding)
	source = sourceFile.readlines()
	target = targetFile.readlines()
	sourceFile.close()
	targetFile.close()
	
	# Bundle them together in tuple
	result = []
	if(len(source) != len(target)):
		printError("Mismatched read size, 2 file: {}-{}".format(len(source), len(target)))
	for i in range(len(source)):
		aligment_parsed = createAlignmentFromCouple(source[i], target[i])
		score = scoreFromAlignmentArray(aligment_parsed)
		result.append((source[i], target[i], aligment_parsed, score))
		
	return result
	
def writeSetToFiles(writeDir, src, tgt, sets, encoding='utf-8'):
	sourceFile = io.open(writeDir + '.' + src, mode='w', encoding=encoding)
	targetFile = io.open(writeDir + '.' + tgt, mode='w', encoding=encoding)
	alignmentFile = io.open(writeDir + '.align', mode='w', encoding=encoding)
	counter = 0
	totalValue = 0
	for set in sets:
		totalValue += set[3]
		counter += 1
		sourceFile.write(set[0])
		targetFile.write(set[1])
		if(PRINT_SCORE.find('on') >= 0):
			alignmentFile.write(u'({})'.format(set[3]) + set[2])
		else:
			alignmentFile.write(set[2])
	if(PRINT_SCORE.find('avg') >= 0 and counter > 0):
		alignmentFile.write("Average: {}".format(totalValue / counter))
	sourceFile.close()
	targetFile.close()
	alignmentFile.close()
	
def checkAndWrite(outputdir, src, tgt, sets, compareValue, differentiator, encoding='utf-8'):
	# Split sets into those pass compareValue and those don't
	larger = [x for x in sets if x[3] >= compareValue]
	smaller = [x for x in sets if x[3] < compareValue]
	
	# Write to files (larger)
	writeSetToFiles(outputdir + differentiator[0], src, tgt, larger, encoding)
	
	# Write to files (smaller)
	writeSetToFiles(outputdir + differentiator[1], src, tgt, smaller, encoding)

def writeTauOnly(outputdir, listResult, extension, writeFoundAlignment=False, encoding='utf-8'):
	alignmentFile = io.open(outputdir + '.' + extension, mode='w', encoding=encoding)
	
	counter = 0
	totalTau = 0.0
	for _, _, alignment, tau in listResult:
		alignmentFile.write("%.5f  " % tau)
		if(writeFoundAlignment):
			alignmentFile.write("\t{}".format(alignment))
		alignmentFile.write("\n")
		totalTau += tau
		counter += 1
	
	alignmentFile.write("Average tau: %.5f, %d sentences." % (totalTau / float(counter), counter))
	
	alignmentFile.close()

if __name__ == "__main__":
	# Run argparse
	parser = argparse.ArgumentParser(description='Scoring lines and separate them.')
	#parser.add_argument('integers', metavar='N', type=int, nargs='+',
	#					help='an integer for the accumulator')
	parser.add_argument('--src', type=str, default='ja', help='source extension')
	parser.add_argument('--tgt', type=str, default='en', help='target extension')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input file')
	parser.add_argument('-o','--outputdir', type=str, default=None, help='location of the output file')
	parser.add_argument('-m','--mode', type=str, default='normal', help='normal mode calculate the tau and split it by the differentiator, compare mode run two files to compare the tau value')
	parser.add_argument('--compare_extension', type=str, default='tau', help='the extension for the tau output file in compare mode, default tau')
	parser.add_argument('-d','--differentiator', type=getDifferentiator, default="passed.failed", help='names for aligments larger|smaller than tau, format larger.smaller')
	parser.add_argument('-t','--tau', type=float, default=0.0, help='tau coefficient for comparison')
	parser.add_argument('--printscore', type=str, default='off', help='print score alongside the alignment')
	parser.add_argument('--print_alignment', action='store_true', help='print the detected alignment alongside tau score during compare mode, default false')
	args = parser.parse_args()
	if(args.outputdir is None):
		args.outputdir = args.inputdir
	global PRINT_SCORE
	PRINT_SCORE	= args.printscore
	
	# Execute script
	if(args.mode == 'normal'):
		sets = readTogether(args.inputdir, args.src, args.tgt)
		checkAndWrite(args.outputdir, args.src, args.tgt, sets, args.tau, args.differentiator)
	elif(args.mode == 'compare'):
		sets = createScore(args.inputdir, args.src, args.tgt)
		writeTauOnly(args.outputdir, sets, args.compare_extension, args.print_alignment)
	else:
		raise argparse.ArgumentTypeError("Incorect mode, must be normal|compare")