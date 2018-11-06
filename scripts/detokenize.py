import sys, io, os, re
from cleaner import add_capitalization, cap_token

def revert_tokenization(line):
  # tokenized line had no caps and 
  pass

def revert_capitalization(tokens):
  # capitalize the next words caught after <cap> token
  new_tokens = []
  prev_cap = False
  for token in new_tokens:
    if(token == cap_token):
      # is cap token, enable flag
      if(prev_cap == True):
        print("Error: two <cap> tokens detected @revert_capitalization")
      prev_cap = True
    else:
      # not cap token, process
      if(cap_token == True):
        # capitalize
        token[0] = token[0].upper()
        cap_token = False
      # append
      new_tokens.append(token)
  return new_tokens

def detokenize(line):
  return re.sub("_", " ", line)

if __name__ == "__main__":
  for file_path in sys.argv[1:]:
    if(not os.path.isfile(file_path)):
      print("Args {:s} not valid file. Skipping.".format(file_path))
    file_path_base, file_path_ext = os.path.splitext(file_path)
    detok_file_path = file_path_base + "_detok" + file_path_ext
    with io.open(file_path, "r", encoding="utf8") as tokenized_file:
      lines = tokenized_file.readlines()
      detokenized_lines = [detokenize(line) for line in lines]
      with io.open(detok_file_path, "w", encoding="utf8") as detokenized_file:
        print("Write {:d} detokenized lines to {:s}".format(len(detokenized_lines), detok_file_path))
        detokenized_file.writelines(detokenized_lines)
