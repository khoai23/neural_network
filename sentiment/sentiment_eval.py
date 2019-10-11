import matplotlib.pyplot as plt
import numpy as np

def createHeatmapForSentence(ax, sentence, attentions, subtitle=None, attention_names=None, cmap="Greys"):
	"""Draw a heatmap anotating the attentions on the sentence on a constructed ax
		Args:
			ax: the pyplot axes to draw the image into
			sentence: the unsplitted sentence, str
			attentions: the attentions that spread on the sentence, [num_att, ?]
			attention_names: the names for the attentions if num_att>1
			subtitle: the title of this particular ax(subplot)
			cmap: the cmap used for drawing
		Returns:
			The image of pyplot drawn on the ax
	"""
#	print(attentions, attention_names)
	if(len(attentions) == 1 or attention_names == None):
		attention_names = ["Attention"]
	assert len(attentions) == len(attention_names), "Mismatch: attention and names have different 1st dimension"
	# draw the heatmap on the sentences
	words = sentence.strip().split()
	attentions = attentions[:, :len(words)+1]
#	print("Attention: {}".format(attentions))
	image = ax.imshow(attentions, cmap=cmap)
	# show only the [attention_type, sentence_length] portion
	ax.set_xticks(np.arange(len(words)))
	ax.set_xticklabels(words)
	ax.set_yticks(np.arange(len(attention_names)))
	ax.set_yticklabels(attention_names)
	if(subtitle):
		ax.set_title(subtitle)
	# return the image
	return image 

def createHeatmapImage(stream, sentences, attentions, scores, image_title=None, attention_names=None, cmap="Greys"):
	"""Draw the image incorporating all the sentences
		Args:
			stream: the stream to put the image into. IOBuffer
			sentences: the unsplitted sentence, [num_sample] str
			attentions: the attentions that spread on the sentence, [num_sample, num_att, ?]
			scores: the list of prediction/correct score, [num_sample, 2]
			image_title: the title of the image, default None
			attention_names: the names for the attentions if num_att>1, [num_att] str
		Return:
			the stream
	"""
	num_sample = len(sentences)
	figure, ax_list = plt.subplots(num_sample, 1, figsize=(6.4, 2.0 * num_sample))
	if(image_title):
		figure.suptitle(image_title, y=1.08)
	iterator = zip(sentences, attentions, ax_list, scores)
	for sentence, attention_s, ax, score in iterator:
		predict, actual = score
		score_str = "Predict {} - Actual {}".format(predict, actual)
		createHeatmapForSentence(ax, sentence, attention_s, attention_names=attention_names, cmap=cmap, subtitle=score_str)
	
	plt.tight_layout(pad=1.0)
	plt.savefig(stream, format="png", bbox_inches='tight', dpi=120)
	plt.close(figure)
	return stream

def createRatingMaps(stream, bundled_eval_data, image_title=None, cmap="Greys"):
	"""Draw a map showing the distributions of evaluation result
		Args:
			stream: the stream to send the image into. IOBuffer
			bundled_eval_data: the data after evaluation, [num_sample] of (corr, line, pred)
			image_title: the header for the map, default None
		Returns:
			the stream with the rating map
	"""
	# assert eval data and all concerning 
	sample_data = np.zeros([5, 5], dtype=int)
	for c, _, p in bundled_eval_data:
		col = int(c) - 1
		row = int(p) - 1
		# row is first; column later
		sample_data[row][col] += 1
#	sample_data = np.asarray(sample_data)
	labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
	text_color = ("black", "white")
	bound_color = ("red", "green")
	# perform drawing using matplotlib
	figure = plt.figure(1)
	ax = plt.gca()
	image = ax.imshow(sample_data, cmap=cmap)
	if(image_title):
		figure.suptitle(image_title, y=1.08)
	# colorbar, the occurence label might be unnecessary
	colorbar = ax.figure.colorbar(image, ax=ax)
	colorbar.ax.set_ylabel("Occurence", rotation=-90, va="bottom")
	# labels
	ax.set_xticks(np.arange(5))
	ax.set_yticks(np.arange(5))
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
	plt.xlabel("Column - Correct")
	plt.ylabel("Row - Prediction")
	# annotation
	color_threshold = image.norm(sample_data.max()) / 2
	kw = dict(horizontalalignment="center", verticalalignment="center")
	sample_text = []
	for i, sample_row in enumerate(sample_data):
		for j, sample_val in enumerate(sample_row):
			# greys means white below threshold and black above, so text is black (0-false) if below and white (1-true) if above
			kw.update(color=text_color[ image.norm(sample_val) > color_threshold ])
			# if is a good (same positive/negative/neutral) box, draw green circle, else draw red
			kw.update(bbox={"facecolor":"none", "edgecolor":bound_color[i == j or (i+1-3)*(j+1-3) > 0], "boxstyle":"circle" })
			text = image.axes.text(j, i, str(sample_val), **kw)
			sample_text.append(text)
	# save to stream
	figure.tight_layout(pad=1.0)
	plt.savefig(stream, format="png", bbox_inches='tight', dpi=120)
	plt.close(figure)
	return stream

_DEFAULT_PLOTLINE_SETTING = [("green", "o"), ("blue", "x"), ("black", "^"), ("red", "s")]
def createDistributionGraph(stream, data, image_title=None, legend_names=None):
	"""Create the distribution graph for the processed data
		Args:
			stream: the stream to put the image into. IOBuffer
			data: the data to show the distribution of, [subset][num_sample] of (line, rating, ...)
			image_title: the title of the image
			legend_names: the names of each sub-dataset in data
		Returns:
			stream with the graph in it
	"""
	assert len(data) <= 4, "Currently cannot support more than 4 items"
	assert len(legend_names) == len(data), "Mismatch names vs dataset: {}-{}".format(len(legend_names), len(data))
	# init figure
	figure = plt.figure(1)
	if(image_title):
		figure.suptitle(image_title, y=1.08)
	# start drawing
	list_lines = []
	for idx, dataset in enumerate(data):
		distribution = {}
		for item in dataset:
			# assume that the rating is in str
			rating = int(item[1])
			distribution[rating] = distribution.get(rating, 0) + 1
		distribution = sorted( distribution.items(), key=lambda item: item[0] )
		distribution_x, distribution_y = zip(*distribution)
		# convert to percentage
		total_case = float(len(dataset))
		distribution_y = [100.0 * float(num_case) / total_case for num_case in distribution_y]
		color, marker = _DEFAULT_PLOTLINE_SETTING[idx]
		# save the line for the legend
		line, = plt.plot(distribution_x, distribution_y, color=color, marker=marker)
		list_lines.append(line)
	# specify the legend, and add it to the top right of the figure
	plt.legend(list_lines, legend_names, loc="upper left", bbox_to_anchor=(1.05, 1))
	# add labels for axis
	plt.xlabel("Rating (1-5)")
	plt.ylabel("Distribution (%)")
	# save to stream
	figure.tight_layout(pad=1.0)
	plt.savefig(stream, format="png", bbox_inches='tight', dpi=120)
	plt.close(figure)
	return stream
