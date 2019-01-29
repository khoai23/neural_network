import sys, io, re, itertools, json, os

tsvToken = re.compile("\d+-\d+\t\d+-\d+\t(.+?)\t(_|\d\[\d+\])")
tsvLine = re.compile("#Text=(.+)")

def readTSVIntoBlocks(lines):
	# split the lines to blocks
	block = []
	for idx, line in enumerate(lines):
		if(line[0] == "#"):
			# record the sentence
			current_line_match = re.match(tsvLine, line.strip())
			current_line = None if current_line_match is None else current_line_match.group(1)
		elif(line.strip() == ""):
			if(len(block) > 0 and current_line is not None):
				# yield the current caught line and the array of (token, rating) that goes with it
				yield current_line, block
			else:
				continue
			# regardless of yielding or not, reset
			block = []
			current_line = None
		else:
			# match and extract words and token
			matcher = re.match(tsvToken, line)
			if(matcher is None):
				print("Line {:s} error, cannot match.".format(line.strip()))
				continue
			else:
				block.append( (matcher.group(1), matcher.group(2)) )
	if(len(block) > 0):
		# flush out the last
		yield current_line, block

def convertBlockToRating(blockGenerator):
	data = []
	for line, block in blockGenerator:
		tokens, ratings = zip(*block)
		ratings = [ int(token_rating[0]) for token_rating in ratings if token_rating != "_"]
		ratings = [k for k, g in itertools.groupby(ratings)]
		if(len(ratings) == 0):
			pass
		elif(len(ratings) == 1):
			data.append( (line, ratings[0]) )
		else:
			data.append( (line, int(sum(ratings) // len(ratings))) )
	return data

if __name__ == "__main__":
	folder_tsv_in = sys.argv[1]
	file_json_out = sys.argv[2]
	print("Read data from tsv files at {:s}".format(folder_tsv_in))
	all_lines = []
	for root, subdirs, files in os.walk(folder_tsv_in):
#		print(root, subdirs, files)
		for file_dir in files:
			if(any ((ext == file_dir[-4:]) for ext in [".tsv", ".txt"])):
				file_dir = os.path.join(root, file_dir)
				print("\tFile {:s} valid, loading in.".format(file_dir))
				with io.open(file_dir, "r", encoding="utf-8") as tsv_file:
					all_lines.append(tsv_file.readlines())
	# convert to list of (line, block)
	raw_blocks = [readTSVIntoBlocks(lines) for lines in all_lines]
	# convert to (line, rating)
	lines_with_ratings = [convertBlockToRating(blocks) for blocks in raw_blocks]
	print("Load data ({} lines) into json file at {:s}".format([len(lines) for lines in lines_with_ratings], file_json_out))
	# flatten and write into format, make sure to convert ratings to str
	lines_with_ratings = (line for lines in lines_with_ratings for line in lines)
	lines, ratings = zip(*lines_with_ratings)
	ratings = [str(rat) for rat in ratings]
	json_blob = [{"lines": lines, "ratings": ratings}]
	with io.open(file_json_out, "w", encoding="utf-8") as json_file:
		json.dump(json_blob, json_file, ensure_ascii=False, indent=4)
	print("Done, json created with {:d} lines".format(len(lines)))
