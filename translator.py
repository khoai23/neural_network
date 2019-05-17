import argparse, io, json, os, sys
from translator_module import DefaultSeq2Seq, check_blank_lines
import tensorflow as tf

max_steps = 50000

with io.open("translator_config.json", "r", encoding="utf-8") as config_file:
	DATA_CONFIG = json.load(config_file)

def files_verification(list_file_paths):
	for path in list_file_paths:
		assert os.path.isfile(path), "Path {:s} is not a valid file!".format(path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Translator using an estimator module")
	parser.add_argument("mode", choices=["train", "eval", "infer", "train_and_eval", "score"], help="The running mode of the script")
	parser.add_argument("--data", choices=DATA_CONFIG.keys(), required=True, help="The data config to be used")
	args = parser.parse_args()
	# unload data paths
	data_config = DATA_CONFIG[args.data]
	vocab_location_en = data_config["vocab_location_en"]
	vocab_location_vi = data_config["vocab_location_vi"]
	train_location_en = data_config["train_location_en"]
	train_location_vi = data_config["train_location_vi"]
	eval_location_en = data_config["eval_location_en"]
	eval_location_vi = data_config["eval_location_vi"]
	model_position = data_config["model_location"]
	# start running
	print("Running mode {}".format(args.mode))
	translation_model = DefaultSeq2Seq(num_units=128, vocab_files=(vocab_location_en, vocab_location_vi), model_dir=model_position)
	# set the verbosity
	translation_model.verbosity(tf.logging.DEBUG)
	if(args.mode == "train"):
		mode = tf.estimator.ModeKeys.TRAIN
		train_input_fn = lambda: translation_model.build_batch_dataset_tensor( (train_location_en, train_location_vi), mode=mode )
		train_hooks = translation_model.model_hook(mode)
		translation_model.estimator.train(input_fn=train_input_fn, hooks=train_hooks, max_steps=max_steps)
		print("Training completed")
	elif(args.mode == "eval"):
		mode = tf.estimator.ModeKeys.EVAL
		# check eval files
		_ = check_blank_lines(eval_location_en), check_blank_lines(eval_location_vi)
		eval_input_fn = lambda: translation_model.build_batch_dataset_tensor( (eval_location_en, eval_location_vi), mode=mode )
		eval_hooks = translation_model.model_hook(mode, eval_metric="bleu", eval_reference_file=eval_location_vi)
		translation_model.estimator.evaluate(input_fn=eval_input_fn, hooks=eval_hooks)
		print("Eval completed")
	elif(args.mode == "infer"):
		mode = tf.estimator.ModeKeys.PREDICT
		predict_input_fn = lambda: translation_model.build_infer_dataset_tensor( eval_location_en )
		predictions = translation_model.estimator.predict(input_fn=predict_input_fn)
		for pred in predictions:
			# format into acceptable tuple
			prediction = (pred["tokens"], pred["ids"], pred["length"])
			translation_model.format_prediction(prediction, sys.stdout)
	else:
		raise NotImplementedError()

