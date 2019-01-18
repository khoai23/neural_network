import sys, io, re

checkerCSVRegex = re.compile("\"(.+)\",\"([\d,]+)\"," + ",".join(["(\d+)"] * 5))

def readCSV(lines):
	# 2 strings, 5 numbers
	# double quote " is converted to double-quote '' in strings
	data = []
	for line in lines:
		matcher = re.match(checkerCSVRegex, line)
		if(matcher is None):
			print("Sentence {:s} do not fit checkerCSVRegex, recheck the code if necessary".format(line.strip()))
			continue
		content = matcher.group(1).replace("\'\'", "\"")
		stars = tuple( (int(val) for val in matcher.group(3, 4, 5, 6, 7)) )
		data.append( (content, stars) )
	return data

def selectStar(examples, sort_by_certainty=True):
	# choose the highest star. On the event of equal star, use disparity (e.g max=2=3 then compare 1 vs 4+5). If fail, chose one lower
	# if there are more than 2 equal stars, select 3(id. 2)
	stared_examples = []
	for content, stars in examples:
		max_star = max(stars)
		max_indices = [i for i, st in enumerate(stars) if st == max_star]
		if(len(max_indices) == 1):
			star_idx = max_indices[0]
		elif(len(max_indices) == 2):
			selector = 0 if sum(stars[:max_indices[0]+1]) >= sum(stars[max_indices[1]:]) else 1
			star_idx = max_indices[selector]
		else:
			star_idx = 2
		certainty = float(stars[star_idx]) / float(sum(stars))
		# correct star is star_idx + 1
		stared_examples.append( (content, star_idx+1, certainty) )
	# sort if selected, from low to high
	if(sort_by_certainty):
		stared_examples = sorted(stared_examples, key=lambda it: it[-1])
	return [(content, star) for content, star, _ in stared_examples]

tsvHeader = "#FORMAT=WebAnno TSV 3.2\n#T_SP=webanno.custom.Sentence_Sentiment|Rating\n"
tsvWordFormat = "\n{:d}-{:d}\t{:d}-{:d}\t{:s}\t{:d}[{:d}]\t"

def convertToTSV(stared_examples, stream):
	# load the format, and custom tag
	stream.write(tsvHeader)
	# write the examples to file
	overall_index = 0
	for content_index, (content, star) in enumerate(stared_examples):
		# sentence separator
		stream.write("\n")
		# first, the old unformatted sentence
		stream.write("\n#Text={:s}".format(content))
		# split the into words, and format them
		word_begin = 0
		word_count = 1
		for char_idx, char in enumerate(content):
			if(char in " .,\"?!"):
				# if word separator or sentence separator
				word_end = char_idx
				if(word_begin < word_end):
					# only add word if the word is not empty(length = 1)
					word = content[word_begin:word_end]
					stream.write(tsvWordFormat.format(content_index+1, word_count, overall_index+word_begin, overall_index+word_end, word, star, content_index+1))
					word_count += 1
				if(char != " "):
					# is sentence separator, add them as a token
					word_begin = char_idx
					word_end = char_idx + 1
					stream.write(tsvWordFormat.format(content_index+1, word_count, overall_index+word_begin, overall_index+word_end, char, star, content_index+1))
					word_count += 1
				# once done, set the begining of the next word to the last end
				word_begin = word_end + 1
		if(word_begin < len(content)):
			# load the last word
			word_end = len(content)
			word = content[word_begin:word_end]
			stream.write(tsvWordFormat.format(content_index+1, word_count, overall_index+word_begin, overall_index+word_end, word, star, content_index+1))
		# done with the whole thing, add to the overall index
		# add content_length and 2 to the overall index since the separator between contents are \n\n
		overall_index += len(content) + 2

CHUNK_SIZE = 1000

if __name__ == "__main__":
	file_csv_in = sys.argv[1]
	file_tsv_out = sys.argv[2]
	print("Read data from csv file at {:s}".format(file_csv_in))
	with io.open(file_csv_in, "r", encoding="utf-8") as file_in:
		raw_content = readCSV(file_in.readlines())
		stared_content = selectStar(raw_content)
	# split to chunk size of 1k
	print("Load data into tsv file at {:s}".format(file_tsv_out))
	if("{:d}" in file_tsv_out):
		print("Have index format, print out as multiple indexed files with chunk size {:d}.".format(CHUNK_SIZE))
		for chunk_idx in range(len(stared_content) // CHUNK_SIZE):
			chunk = stared_content[CHUNK_SIZE*chunk_idx : CHUNK_SIZE*(chunk_idx+1)]
			with io.open(file_tsv_out.format(chunk_idx+1), "w", encoding="utf-8") as file_out:
				convertToTSV(chunk, file_out)
	else:
		print("No index format, print out as one file.")
		with io.open(file_tsv_out, "w", encoding="utf-8") as file_out:
			convertToTSV(stared_content, file_out)
	print("Done.")
