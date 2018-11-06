from collections import Counter
import sys, io, os
if __name__ == "__main__": 
  list_path = [path for path in sys.argv[1:] if os.path.isfile(path)]
  try:
    threshold = int(sys.argv[1])
    if(threshold < 0):
      print("Found threshold {:d} (bound)".format(threshold))
    else:
      print("Found threshold {:d} (threshold)".format(threshold))
  except ValueError:
    print("Warning! you have not specified the threshold. The vocab will contain all words at this rate.")
    threshold = 0
  for path in list_path:
    with io.open(path, "r", encoding="utf8") as read_file:
      word_count = Counter(read_file.read().split())
    total_words_count = len(word_count.items())
    words_num = sum(word_count.values())
    if(threshold >= 0):
      words = [(word, counter) for word, counter in sorted( word_count.items(), key=lambda item: item[0] ) if counter >= threshold]
    else:
      bound = -threshold
      # from largest to smallest, so reverse
      words = list(sorted(word_count.items(), key=lambda x: x[1], reverse=True))[:bound]
    words, counters = zip(*words)
    covered_num = sum(counters)
    true_path, extension = os.path.splitext(path)
    vocab_path = true_path + "_vocab" + extension
    with io.open(vocab_path, "w", encoding="utf8") as vocab_file:
      vocab_file.writelines("\n".join(words))
    print("Done generating vocab, saved at {:s}, words counted {:d}, total words {:d}, covering {:d}/{:d}({:.2f}%)".format(vocab_path, len(words), total_words_count, covered_num, words_num, float(covered_num) / float(words_num) * 100.0))
    del words
