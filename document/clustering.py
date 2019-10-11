import numpy as np
import sklearn.cluster as cluster
import sklearn.mixture as mixture
import matplotlib.pyplot as plt
import pickle, io, sys, argparse

# import the read data here
def readDataFromPickle(pickleFile):
	data = pickle.load(pickleFile)
	# if tuple of two, return immediately
	if(isinstance(data, tuple) and len(data) == 2):
		return data
	# if one dict, split it into WordDict and matrix
	if(isinstance(data, dict)):
		wordDict = {}
		embeddings = []
		for word in data:
			wordDict[word] = len(embeddings)
			embeddings.append(data[word])
		# make sure the embeddings is 2d matrix
		embeddings = np.asarray(embeddings)
		return wordDict, embeddings

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Do clustering by scikit-learn library.')
	parser.add_argument('-i','--inputdir', type=str, default=None, required=True, help='location of the input file')
	parser.add_argument('-m','--mode', type=str, default=None, required=True, help='cluster mode to be used, must be in sklearn.cluster module')
	parser.add_argument('--cluster_number', type=int, default=10, help='amount of clusters, only for some mode')
	parser.add_argument('-v', '--verbose', type=int, default=0, help='verbosity level')
	parser.add_argument('-o', '--outputdir', type=str, default=None, help='location of output file')
	parser.add_argument('-t', '--output_type', type=str, default='txt', help='type of output, default txt')
	parser.add_argument('--dbscan_eps', type=float, default=0.3, help='dbscan: distance to be consider neighbors, default 0.3')
	parser.add_argument('--dbscan_metric', type=str, default="euclidean", help='dbscan: the type of metric to calculate distances between points')
	parser.add_argument('--birch_maximum_branch', type=int, default=50, help='birch: maximum of cluster in each node')

	args = parser.parse_args()
	if(args.outputdir is None):
		args.outputdir = args.inputdir + ".out"
		print("-o (--outputdir) option missing, default to {:s}".format(args.outputdir))
	
	# read from binary pickle file
	with io.open(args.inputdir, "rb") as pickleFile:
		wordDict, embeddings = readDataFromPickle(pickleFile)
	print("Data read.")
	
	# use scikit lib to compute clusters
	if(args.mode == "kmean" or args.mode == "k-mean"):
		clusterObject = cluster.KMeans(n_clusters=args.cluster_number, verbose=args.verbose)
		labels = clusterObject.fit_predict(embeddings)
	elif(args.mode == "affinity"):
		clusterObject = cluster.AffinityPropagation()
		labels = clusterObject.fit_predict(embeddings)
	elif(args.mode == "agg_ward" or args.mode == "agg_complete" or args.mode == "agg_average"):
		clusterObject = cluster.AgglomerativeClustering(n_clusters=args.cluster_number, linkage=args.mode[4:])
		labels = clusterObject.fit_predict(embeddings)
	elif(args.mode == "dbscan"):
		print("Using DBSCAN with eps {:f}, metric {:s}".format(args.dbscan_eps, args.dbscan_metric))
		clusterObject = cluster.DBSCAN(eps=args.dbscan_eps, metric=args.dbscan_metric)
		labels = clusterObject.fit_predict(embeddings)
	elif(args.mode == "birch"):
		clusterObject = cluster.Birch(n_clusters=args.cluster_number, branching_factor=args.birch_maximum_branch)
		labels = clusterObject.fit_predict(embeddings)
	elif(args.mode == "gauss"):
		raise NotImplementedError("Gaussian currently not working. Use another mode.")
		gaussMixture = mixture.GaussianMixture
	else:
		raise argparse.ArgumentError(message="Mode not recognized: {:s}".format(args.mode))
	print("Clustering completed.")
	
	if(args.output_type == 'txt'):
		# print to file the respective group
		with io.open(args.outputdir, "w", encoding='utf8') as writeFile:
			floorValue = max(np.min(labels), 0)
			numGroup = max(np.max(labels) - floorValue, 0) + 1
			groupList = [[] for i in range(numGroup)]
			print(floorValue, numGroup)
			for word in wordDict:
				groupIdx = labels[wordDict[word]] - floorValue
				groupList[groupIdx].append(word)
#			print(len(groupList))
			# after grouping manually, write to file
			for idx, group in enumerate(groupList):
				groupSize = len(group)
				groupString = ", ".join(group)
				print(idx, groupSize)
				writeFile.write("Group {:d}, number of members {:d}: {:s}\n\n".format(idx+1, groupSize, groupString))
	elif(args.output_type == 'binary'):
		with io.open(args.outputdir, "wb") as binaryWriteFile:
			pickle.dump((wordDict, embeddings, labels))
	else:
		raise argparse.ArgumentError(message="Output type not recognized: {:s}".format(args.output_type))





def k_mean_clustering(listPoints, numCluster, stopThreshold=None, debugFn=None, maximumIterations=20000):
	return NotImplementedError("Try later after running with sklearn")
	# apply the k-mean clustering here
	assert isinstance(numCluster, int) and numCluster > 1, "Invalid numCluster (must be >1 int): {}".format(numCluster)
	
	# initialize random length 1 vector for the cluster point
	vectorSize = np.shape(listPoints)[2]
	clusterHeads = np.random.normal(size=(numCluster, vectorSize))
	clusterIniialSize = np.linalg.norm(clusterHeads)
	clusterHeads = np.divide(clusterHeads, clusterIniialSize)
	
	# initiate for the while value
	# if stopThreshold not initialized, base it on an abitrary small number since theoretically it will always converge on full batch
	if(stopThreshold is None):
		stopThreshold = 1e-10 * numCluster
	totalVariance = stopThreshold + 0.1
	iterCounter = 0
	while(totalVariance > stopThreshold and iterCounter < maximumIterations):
		iterCounter += 1
		totalVariance = 0.0
		# affiliate the points based on the nearest clusterHeads
		# use dot product, then argmax to get the indexes
		closestHeads = np.matmul(listPoints, np.transpose(clusterHeads))
		closestHeads = np.argmax(closestHeads)
		# brute: iterate through the first dimension and gather them by closestHeads
		counter = [0] * numCluster
		sumVectors = np.zeros(np.shape(clusterHeads))
		for headIdx, point in zip(closestHeads, listPoints):
			sumVectors[headIdx] = np.sum([sumVectors[headIdx], point])
			counter[headIdx] += 1
		# gather the mean vector from the sum by counter
		for i, count, sumVector in enumerate(zip(counter, sumVectors)):
			if(count > 0):
				# replace the head with the mean only if the count is large enough
				oldCluster = clusterHeads[i]
				clusterHeads[i] = sumVector / count
				variance = np.sum(clusterHeads - oldCluster)
				totalVariance += variance
				if(debugFn):
					debugFn("Move cluster head {:d} from {} to {}, distance {:.4f}.".format(i+1, oldCluster, clusterHeads[i], variance))
			else:
				if(debugFn):
					debugFn("Unmoved cluster head {:d}.".format(i+1))
	# return the clusterHeads and the respective range when done
	closestHeads = np.matmul(listPoints, np.transpose(clusterHeads))
	closestHeads = np.argmax(closestHeads)
	return clusterHeads, closestHeads
