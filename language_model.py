import tensorflow as tf
import io
import argparse

START_OF_SENTENCE_TOKEN = "<s>"
END_OF_SENTENCE_TOKEN = "<\s>"

def convertSentenceToSkipgram(line, window_size=2):
	# assume line is properly tokenized already
	tokens = [START_OF_SENTENCE_TOKEN] * window_size + line.strip().split() + [END_OF_SENTENCE_TOKEN] * window_size
	full_window_size = 2 * window_size + 1
	[ tokens[i:i+full_window_size] for i in range(len(tokens) - window_size * 2)]


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Language Models.')
	parser.add_argument('-m', '--model_type', type=str, choices=["skipgram", "whole"], default="skipgram", help='The type of model used')
	return parser.parse_args()
