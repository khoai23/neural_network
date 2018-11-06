import sys, io, os, re
import cleaner

COUNTER_PRINT = 10000

# this is for non-standard clearing set
replacement_set = [("–", "-")]

symbols = "!@#\$%\^\*\(\)\+={}\|\[\]\/\?:;<>\'\"\,\.\-"
spacer = re.compile("[{:s}]".format(symbols))
space_reductor = re.compile(" {2,}")
def clean(line):
  # add spaces for all special characters
  line = re.sub(spacer, " \\g<0> ", line)
  # reduce all many to one space
  line = re.sub(space_reductor, " ", line)
  # strip here to prevent odd cases
  line = line.strip()
  return line

if __name__ == "__main__":
  if(os.path.isfile(sys.argv[1])):
    print("Using no mode.")
    mode = ""
  else:
    mode = sys.argv[1]
    if("remove" in mode):
      print("Will remove number + tab at the start of the sentences.")
    if("cap" in mode):
      print("Will change all capitalized word into <cap> + word, eg. Jane -> <cap> jane")
    if("lower" in mode):
      print("Will indiscriminately lowercase everything")
    if("wiki" in mode):
      print("Will try to clean up on wikipedia crawler format")
    if("vi" in mode):
      print("Will remove external diacritics")
    print("Mode: {:s}".format(mode))
  file_in_dir = sys.argv[2]
  file_out_dir = sys.argv[3]
  # Do everything.
  with io.open(file_in_dir, "r", encoding="utf8") as uncleaned_file, io.open(file_out_dir, "w", encoding="utf8") as cleaned_file:
  #  lines = uncleaned_file.readlines()
    # vi diacritics preparation
    if("vi" in mode):
      all_ghost_diacritics, convert_dict = cleaner.generate_tranform_dict()
      diacritic_regex = re.compile("[{:s}]".format(all_ghost_diacritics))
    print("Dict: {}".format(convert_dict))
    # counter preparation
    global counter
    counter = 0
    # main function
    def process_line(orig_line):
      # add boredom-resistant counter
      global counter
      counter += 1
      if(counter % COUNTER_PRINT == 0):
        print("Lines processed: {:d}".format(counter))
      # detokenize things
      if(not orig_line.strip()):
        # blank, output to stdout and return it without \n
        print("Empty line detected @{:d}".format(counter))
        return ""
      try:
        if("vi" in mode):
          orig_line = cleaner.vietnamese_ghost_characters_cleaner(orig_line, all_ghost_diacritics, convert_dict, ghost_checker_regex=diacritic_regex, debug_line_number=counter)
        # wiki mode
        line = clean(orig_line)
        if("wiki" in mode):
          line = cleaner.replace_model_token(line)
          line = cleaner.remove_milestone(line, tokenized=True)
          line = cleaner.remove_first_number(line)
          line = cleaner.replace_number_tokens(line, tokenized=True)
        if("wiki" in mode):
          line = cleaner.rejoin_middle_name(line)
          line = cleaner.remove_image_tokens(line)
        # decap or lower if specified
        if("cap" in mode):
          tokens = line.split()
          tokens = cleaner.add_capitalization(tokens, consider_all_caps=True)
          line = " ".join(tokens)
        elif("lower" in mode):
          line = line.lower()
      except Exception as e:
        print("Exception caught @ line {:d}-{:s}".format(counter, orig_line))
        raise e
      return line.strip() + "\n"
    cleaned_file.writelines((process_line(line) for line in uncleaned_file.readlines()))
  print("Cleaning completed, exported from {:s} to {:s}".format(file_in_dir, file_out_dir))
