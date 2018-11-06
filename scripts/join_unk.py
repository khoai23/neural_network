import sys, io, os

def join_unk(line, vocab):
  tokens = line.strip().split()
  new_tokens = []
  prev_unk = None
  for token in tokens:
    if(token in vocab):
      # is a valid token, add it with the prev_unk (if exist)
      if(prev_unk is not None):
        assert isinstance(prev_unk, str), "{}: unk not string?".format(prev_unk)
        new_tokens.append(prev_unk)
        prev_unk = None
      new_tokens.append(token)
    else:
      # is an unknown word, add to prev_unk
      if(prev_unk is None):
        # new word
        prev_unk = token
      else:
        # append
        prev_unk = prev_unk + "_" + token
  return " ".join(new_tokens)

if __name__ == "__main__":
  target_file_path, vocab_file_path = sys.argv[1:3]
  target_base, target_ext = os.path.splitext(target_file_path)
  processed_file_path = target_base + "_joined_unk" + target_ext
  with io.open(vocab_file_path, "r", encoding="utf8") as vocab_file:
    print("Read vocab file.")
    vocab = set(token.strip() for token in vocab_file.readlines())
  with io.open(target_file_path, "r", encoding="utf8") as target_file:
    print("Read target file and process to joined unk.")
    lines = target_file.readlines()
    # this process strip the line, so need rejoining with \n
    joined_unk_lines = [join_unk(line, vocab) for line in lines]
  with io.open(processed_file_path, "w", encoding="utf8") as write_file:
    print("Write data to {:s}".format(processed_file_path))
    write_file.write("\n".join(joined_unk_lines))

