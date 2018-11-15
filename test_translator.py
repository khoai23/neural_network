import sys, io, os
import translator_clean as test_obj
import numpy as np

def testPaddingRandom():
	dummy_src_len = np.random.randint(1, 50, size=100)
	dummy_tgt_len = np.random.randint(1, 50, size=100)
	dummy_src = [np.random.randint(0, 4000, size=src_len).tolist() for src_len in dummy_src_len]
	dummy_tgt = [np.random.randint(0, 4000, size=tgt_len).tolist() for tgt_len in dummy_tgt_len]
	dummy_dataset = zip(dummy_src, dummy_tgt, dummy_src_len, dummy_tgt_len)
	print("Dummy data generated: {}".format(dummy_dataset))

	result = test_obj.batchAndPad(dummy_dataset, 10, padding_idx=1)
	for idx, item in enumerate(result):
		print("Batch {}, batch content: {}".format(idx, item))

if __name__ == "__main__":
	testPaddingRandom()
