import json, io, os, sys
from sentiment_data import reduceData

def reliability_score(score_dict):
	score_counts = score_dict.values()
	total_occurence = float(sum(score_counts))
	max_occurence = float(max(score_counts))
	return (0.75 - max_occurence) / total_occurence

def run_manual_rating(json_filtered_file, json_cleaned_file):
	with io.open(json_filtered_file, "r", encoding="utf-8") as comment_file:
		comments = json.load(comment_file)
		if(isinstance(comments, dict)):
			print("Data saved as split; load combined")
			iterable_comments = comments["eval"] + comments["train"]
		elif(isinstance(comments, list)):
			print("Data saved as list; load as-is")
			iterable_comments = comments
		else:
			raise ValueError("Filtered comments load error; please recheck the file")
	# this return a dict of (comments, dict_of_ratings)
	comment_counters = { k:v for k,v in reduceData(iterable_comments) }
	# check if output file already exist; if true, load it and remove those already done
	if(os.path.isfile(json_cleaned_file)):
		print("Found previous manual file; loading it...")
		with io.open(json_cleaned_file, "r", encoding="utf-8") as output_file:
			manual_scores, comments = zip( *(line.strip().split("|", 1) for line in output_file.readlines()) )
			for comment in comments:
				comment_counters.pop(comment, None)
	# sort the output by reliability - skewed scores and high occurence get pushed to the back
	comment_sorted_values = sorted(comment_counters.items(), key=lambda item: reliability_score(item[1]), reverse=False)
	# open append operation and output to screen
	with io.open(json_cleaned_file, "a", encoding="utf-8") as output_file:
		print("Enter to select the highest option")
		for comment, score_dict in comment_sorted_values:
			score = None
			score = input("Rate comment: {:s} [score_dict {}]: ".format(comment, score_dict))
			if(score.strip() == ""):
				score = max(score_dict.items(), key=lambda item: item[1])[0]
			output_file.write("{:s}|{:s}\n".format(score, comment))
if __name__ == "__main__":
	if(sys.argv[1] == "help"):
		print("arg1 for input file, arg2 for output file")
		sys.exit(0)
	
	json_filtered_file = sys.argv[1]
	json_cleaned_file = sys.argv[2]
	assert os.path.isfile(json_filtered_file), "arg1 must point to a filtered json file"
	if(json_cleaned_file == "extract"):
		print("Switching to extract mode!")
		with io.open(json_filtered_file, "r", encoding="utf-8") as comment_file:
			comments = json.load(comment_file)
			if(isinstance(comments, dict)):
				print("Data saved as split; load combined")
				iterable_comments = comments["eval"] + comments["train"]
			elif(isinstance(comments, list)):
				print("Data saved as list; load as-is")
				iterable_comments = comments
			else:
				raise ValueError("Filtered comments load error; please recheck the file")
		# split the file into 1k chunk
		output_file_prefix = os.path.splitext(json_filtered_file)[0]
		for i in range(len(iterable_comments) // 1000):
			with io.open("{:s}_{:d}.txt".format(output_file_prefix, i+1), "w", encoding="utf-8") as output_file:
				for comment, _ in iterable_comments[i*1000:(i+1)*1000]:
					comment = comment.strip()
					if(comment == ""):
						continue
					if(comment[-1] in ".!?\"\'"):
						output_file.write(comment + "\n\n")
					else:
						output_file.write(comment + " .\n\n")
		sys.exit(0)
	else:
		print("Doing manual work")
		run_manual_rating(json_filtered_file, json_cleaned_file)



