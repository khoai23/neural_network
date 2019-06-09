import io, argparse, os, sys
from contextlib import ExitStack
from nltk.translate import bleu_score

def score_bleu_tokens(candidate, references):
	candidate = candidate.strip().split()
	references = [ref.strip().split() for ref in references]
	return bleu_score.sentence_bleu(references, candidate)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Align sentences from two source, using translation as references")
	parser.add_argument("-s", "--source", required=True, type=str, help="Source data")
	parser.add_argument("-t", "--target", required=True, type=str, help="Target data")
	parser.add_argument("-r", "--reference", required=True, nargs="*", type=str, help="Reference data")
	parser.add_argument("--references_per_sentence", type=int, default=1, help="The number of references sentences per source sentences ")
	parser.add_argument("--search_window", type=int, default=100, help="The window to search for the next pairing")
	parser.add_argument("--threshold", type=float, default=30.0, help="The threshold to consider two sentences paired")
	parser.add_argument("--enough_threshold", type=float, default=80.0, help="The threshold to instantly consider two sentences paired")
	parser.add_argument("-o", "--output", type=str, default=None, help="The output location")

	args = parser.parse_args()
	files = [args.source, args.target] + args.reference
	for f in files:
		assert os.path.isfile(f), "File {:s} does not exist"
	
	with ExitStack() as stack:
		opened_files = [stack.enter_context(io.open(f, "r", encoding="utf-8")) for f in files]
		# the first 3 is guaranteed, the 4th is not. is source, target, target_ref, {source_ref}
		source, target, target_ref = [f.readlines() for f in opened_files[:3]]
		if(len(opened_files) > 3):
			print("Currently not implemented for source_ref")
		# bundle the references 
		ref_per_sent = args.references_per_sentence
		target_ref = [target_ref[i*ref_per_sent:(i+1)*ref_per_sent] for i in range(len(source))]
		# assume the sentences are not disorganized.
		past_pairing_index = 0
		list_sentence_pairing = []
		for i, (src, tgt_ref) in enumerate(zip(source, target_ref)):
			best_match_score = 0.0
			for tgt_idx in range(past_pairing_index, min(past_pairing_index+args.search_window, len(target))):
				tgt = target[tgt_idx]
				# check with the target
				score = score_bleu_tokens(tgt, tgt_ref) * 100.0
				if(score > best_match_score):
					best_tgt_idx, best_match_score = tgt_idx, score
					if(score > args.threshold):
						# very close sentence, immediately accept
						break
			# acceptable sentence
			if(score > args.threshold):
				list_sentence_pairing.append( (score, src, target[best_tgt_idx]) )
				past_pairing_index = best_tgt_idx
		print("Number of pairings retrieved: {:d}".format(len(list_sentence_pairing)))
		if(args.output is not None):
			output_stream = stack.enter_context(io.open(args.output, "w", encoding="utf-8"))
		else:
			output_stream = sys.stdout
			args.output = "default stream"
		for score, src, tgt in list_sentence_pairing:
			output_stream.write("{:2.2f}\n{:s}\n{:s}".format(score, src.strip(), tgt.strip()))
		print("All done, data written to {:s}".format(args.output))
