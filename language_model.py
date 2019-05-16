#import tensorflow as tf
import data_utils as utils
import tensorflow as tf
import os
import argparse
from bi_rnn_lm import BiRNNLanguageModel

WORD_FILE_LOCATION = "/home/quan/Workspace/Source/translator/txt/Viet74K.txt"
WORD_RELATION_DUMP = "/home/quan/Workspace/Source/translator/txt/xer_distance.txt"


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Language Models.')
	parser.add_argument('phase', type=str, choices=["train", "infer"], help="?")
	parser.add_argument('-m', '--model_type', type=str, choices=["bidir"], default="bidir", help='The type of model used')
	parser.add_argument('--data', type=str, required=True, help="Location of monolingual data")
	parser.add_argument('--size', type=int, default=128, help="Size of hidden/embeddings within model")
	args = parser.parse_args()
	
	# process data
	assert os.path.isfile(args.data)
	word_dict = utils.build_word_dict(args.data)
	dataset = utils.build_dataset(args.data)

	with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:
		# build model
		model = BiRNNLanguageModel(len(word_dict), embedding_size=args.size)
		# create training tensors
		model.prepare_training()
		# train for whole dataset
		for train_batch in utils.batch_iter(dataset, 128, 10):
			model.train(session, train_batch)

#	with io.open(WORD_FILE_LOCATION, "r", encoding="utf-8") as word_file:
#		word_set = loadWords(word_file)
#		print("Word set loaded with {:d} elements".format(len(word_set)))
	
#	word_relation = getCloselyRelatedWords(word_set, word_list=["toi", "chao", "ban"], relate_range=5)

