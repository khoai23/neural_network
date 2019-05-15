import argparse
from translator_module import DefaultSeq2Seq
import tensorflow as tf

train_location_en = "/home/quan/Workspace/Data/iwslt15/train.filter.en"
train_location_vi = "/home/quan/Workspace/Data/iwslt15/train.filter.vi"

eval_location_en = "/home/quan/Workspace/Data/iwslt15/tst2012.en"
eval_location_vi = "/home/quan/Workspace/Data/iwslt15/tst2012.vi"

vocab_location_en = "/home/quan/Workspace/Data/iwslt15/vocab.en"
vocab_location_vi = "/home/quan/Workspace/Data/iwslt15/vocab.vi"

model_position = "/home/quan/Workspace/Data/model/translator/test_run"
max_steps = 50000

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Translator using an estimator module")
	parser.add_argument("mode", choices=["train", "eval", "infer", "train_and_eval", "score"], help="The running mode of the script")
	args = parser.parse_args()

	print("Running mode {}".format(args.mode))
	translation_model = DefaultSeq2Seq(num_units=128, vocab_files=(vocab_location_en, vocab_location_vi), model_dir=model_position)
	# set the verbosity
	translation_model.verbosity(tf.logging.INFO)
	if(args.mode == "train"):
		train_input_fn = lambda: translation_model.build_batch_dataset_tensor( (train_location_en, train_location_vi) )
		train_hooks = translation_model.model_hook(tf.estimator.ModeKeys.TRAIN)
		translation_model.estimator.train(input_fn=train_input_fn, hooks=train_hooks, max_steps=max_steps)
		print("Training completed")
	else:
		raise NotImplementedError()

