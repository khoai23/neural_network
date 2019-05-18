import tensorflow as tf
import subprocess
import os, sys, io

class PredictionMetricHook(tf.train.SessionRunHook):
	"""A hook that retrieve the prediction result and compare it using external scripts
		Should only work on evals 
	"""
	def __init__(self, print_fn, script, clean_after_hook=False, summary_extraction_fn=None, model_dir="./"):
		"""Args:
			print_fn: receive (tokens, length, stream) to write the data
			script: the external command called to evaluate the data
			clean_after_hook: if true, remove the file after evaluation
			summary_extraction_fn: None or callable that receive the result of the script and output (name, scalar) to write on a tensorboard
		"""
		self._script = script
		self._print_fn = print_fn
		self._summary_extraction_fn = summary_extraction_fn
		self._clean = clean_after_hook
		self._model_dir = model_dir
		self._first_write = True

	def begin(self):
		"""Create the subtitute SessionRunContext and leave it there"""
		data = [tf.get_collection("prediction_tokens"), tf.get_collection("prediction_length"), tf.train.get_global_step()]
		self._run_context = data
	
	def before_run(self, run_context):
		"""Append the old run context with the new one"""
#		print("Old session args: ", run_context.original_args.fetches)
		session_args = run_context.original_args
		session_args.fetches.extend(self._run_context)
		return session_args
	
	def after_run(self, run_context, run_values):
		"""Write the batches into the file one batch at a time"""
		prediction_tokens, prediction_length, global_step = run_values.results[-3:]
		self._file_path = file_path = os.path.join(self._model_dir, "prediction.{:d}.txt".format(global_step))
		self._global_step = global_step
		if(self._first_write):
			# check file does not exist, so 'a' mode is justified
			if(os.path.isfile(file_path)):
				tf.logging.warn("Error: file {:s} already existed, removing for new evaluation".format(file_path))
				os.remove(file_path)
			self._first_write = False
		# write for each lines per batch
		with io.open(file_path, "a", encoding="utf-8") as pred_file:
			for batch_tokens, batch_length in zip(prediction_tokens, prediction_length):
				for tokens, length in zip(batch_tokens, batch_length):
					self._print_fn( (tokens, None, length), pred_file )

	def end(self, session):
		"""Run the script to evaluate and broadcast its values"""
		process_command = self._script.format(prediction_file=self._file_path)
		tf.logging.warn("Prediction metric calculation command: {:s} using shell, fix later by subprocess PIPE".format(process_command))
		process_result = subprocess.check_output([process_command], shell=True)
		process_result = process_result.decode("utf-8").strip()
		tf.logging.info("Evaluation output: {:s}".format(process_result))
		if(self._summary_extraction_fn is not None):
			# extra script to extract value and view it on the tensorboard
			summary_name, summary_value = self._summary_extraction_fn(process_result)
			summary = tf.Summary(value=[tf.Summary.Value(tag=summary_name, simple_value=summary_value)])
			summary_writer = tf.summary.FileWriter(self._model_dir)
			summary_writer.add_summary(summary, global_step = self._global_step)
