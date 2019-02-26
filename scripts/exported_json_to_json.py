import sys, io, json, os #, re, itertools
import zipfile
from tsv_to_json import readTSVIntoBlocks

def handleExportedJSON(text_block, references):
	text_block = text_block.replace("\\n", "\n")
	list_sentences = []
	last_offset, last_offset_pos = 0, 0
	for ref in references:
		begin_idx = ref.get("begin", 0)
		end_idx = ref.get("end", -1)
		rating = ref.get("Rating", 3)
		# decrease offset for chars starting @ U+10000
		last_offset += sum( (int(ord(char) >= 0x10000) for char in text_block[last_offset_pos:begin_idx]) )
		begin_idx -= last_offset
		end_idx -= last_offset
		last_offset += sum( (int(ord(char) >= 0x10000) for char in text_block[begin_idx:end_idx]) )
		sentence = text_block[begin_idx:end_idx]
		last_offset_pos = end_idx
		# cut out the sentence
		list_sentences.append((sentence, rating) )
	return list_sentences

def _block_to_idx(block):
	# block containing the indices, so max/min them
	front_ids, back_ids = zip(*block)
	return min(front_ids), max(back_ids)

def backReference(references, original_tsv_file):
	idx_read_fn = lambda matcher_obj: (int(matcher_obj.group(1)), int(matcher_obj.group(2)))
	tsv_sentences = readTSVIntoBlocks(original_tsv_file, read_fn=idx_read_fn)
	references = [ (ref.get("begin", 0), ref.get("end", -1), ref.get("Rating")) for ref in references ]
	tsv_sentences_and_refs = [ (line, _block_to_idx(block)) for line, block in tsv_sentences ]
	list_sentences = []
	for line, (begin, end) in tsv_sentences_and_refs:
		# find a matching rating in references
		rat = next( (rat for b, e, rat in references if begin == b or end == e ), None )
		if(rat is not None):
#			print("Found line: {}, {}".format(line, rat))
			list_sentences.append( (line, rat) )
	return list_sentences

if __name__ == "__main__":
	folder_tsv_in = sys.argv[1]
	file_json_out = sys.argv[2]
	folder_back_reference = None if len(sys.argv) < 4 else sys.argv[3]
	print("Read data from folder at {:s}, backreference specified: {}".format(folder_tsv_in, folder_back_reference))
	data = []
	for root, subdirs, files in os.walk(folder_tsv_in):
		print(root, subdirs, files)
		for file_dir in files:
		# unzip everything 
			if(file_dir[-4:] == ".zip"):
				file_dir = os.path.join(root, file_dir)
				print("Extracting file zip: {:s}".format(file_dir))
				with zipfile.ZipFile(file_dir, "r") as zip_ref:
					# extract at that very location
					zip_ref.extractall(root)

	for root, subdirs, files in os.walk(folder_tsv_in):
		for file_dir in files:
			if(file_dir[-5:] == ".json"):
				# read the json file 
				file_dir = os.path.join(root, file_dir)
				print("\tFile {:s} valid, loading in.".format(file_dir))
				with io.open(file_dir, "r", encoding="utf-8") as json_file:
					json_data = json.load(json_file)
					text_block = list(json_data["_referenced_fss"].values())[0]["sofaString"]
					references = json_data["_views"]["_InitialView"]["Sentence_Sentiment"]
					if(folder_back_reference is None):
						data.append( (text_block, references, None) )
					else:
						document_name = json_data["_views"]["_InitialView"]["DocumentMetaData"][0]["documentTitle"]
						# search for the document within the reference folder
						back_reference_doc = os.path.join(folder_back_reference, document_name)
						if(os.path.isfile(back_reference_doc)):
							print("Found backreference file @{:s}".format(back_reference_doc))
							with io.open(back_reference_doc, "r", encoding="utf-8") as br_file:
								data.append( (text_block, references, br_file.readlines()) )
						else:
							data.append( (text_block, references, None) )
	# do as normal for files without backreferences, but fix them if there are
	lines_with_ratings = [ handleExportedJSON(text_block, references) if(backreference is None) else backReference(references, backreference)
					for text_block, references, backreference in data]
	print("Load data ({} lines) into json file at {:s}".format([len(lines) for lines in lines_with_ratings], file_json_out))
	# flatten and write into format, make sure to convert ratings to str
	lines_with_ratings = (line for lines in lines_with_ratings for line in lines)
	lines, ratings = zip(*lines_with_ratings)
	ratings = [str(rat) for rat in ratings]
	json_blob = [{"lines": lines, "ratings": ratings}]
	with io.open(file_json_out, "w", encoding="utf-8") as json_file:
		json.dump(json_blob, json_file, ensure_ascii=False, indent=4)
	print("Done, json created with {:d} lines".format(len(lines)))
