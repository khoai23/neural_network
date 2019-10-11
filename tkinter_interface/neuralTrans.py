import sys, io, os

def selectModelFolder(folder, log_func=None, raise_error=True):
	if(os.path.isdir(folder)):
		return folder
	else:
		message = "The path specified {:s} is not a folder".format(folder)
		if(raise_error):
			raise Exception(message)
		else:
			log_func(message)
			return None

def runModelFromFolder(folder, data):
	"""Assume we can get the model's config from within the folder itself"""
	raise NotImplementedError("")
