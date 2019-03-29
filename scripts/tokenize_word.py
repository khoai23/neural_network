import sys, io, os, re

COUNTER_PRINT = 10000

# this is for non-standard clearing set
# not yet implemented
replacement_set = [("–", "-")]

symbols = "!@#\$%\^\*\(\)\+={}\|\[\]\/\?:;<>\'\"\,\.\-\n"
spacer = re.compile("[{:s}]".format(symbols))
space_reductor = re.compile(" {2,}")
double_quote = re.compile("``|\'\'")
def clean(line, debug=False):
  if(debug):
    orig_line = line
  # additional cleaning process that most likely won't affect anything
  line = line.replace("-LRB-", "(").replace("-RRB-", ")")
  # elipsis should be spaced by itself, just to be sure
  line = line.replace("...", " \u2026 ")
  line = re.sub(double_quote, "\"", line)
  # add spaces for all special characters
  line = re.sub(spacer, " \\g<0> ", line)
  # reduce all many to one space
  line = re.sub(space_reductor, " ", line)
  # strip here to prevent odd cases
  line = line.strip()
  if(debug):
    print("Clean line: {} -> {}".format(orig_line, line))
  return line

if __name__ == "__main__":
  # put the cleaner here so as to not to intefere with outsider call
  import cleaner
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
    if("ch" in mode):
      print("Will tokenize for chinese characters")
    if("force" in mode):
      print("Will ignore caught exception. Right now there is only vi cleaner using it.")
    if("html" in mode):
      print("Will remove all caught html entities (&..;) and replace with correct defs ")
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
        if("html" in mode):
          orig_line = cleaner.replace_html_entity(orig_line)
        if("vi" in mode):
#          orig_line = cleaner.swap_diacritics(orig_line)
          orig_line = cleaner.vietnamese_ghost_characters_cleaner(orig_line, all_ghost_diacritics, convert_dict, ghost_checker_regex=diacritic_regex) #, debug_line_number=counter
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
        elif("ch" in mode):
          line = " ".join(cleaner.split_line_chinese(line))
      except KeyError as e:
        print("Diacritic error (KeyError) caught @ line {:d}-{:s}".format(counter, orig_line))
        if("force" in mode):
          error_message = str(e)
          print("Additional Error message: {:s}".format(error_message))
          if(len(error_message) == 1):
            print("Genuine KeyError: {:s}({}), returning None".format(error_message, ord(error_message)))
            raise e
          return None
        else:
          raise e
      return line.strip() + "\n"
    process_generator = (process_line(line) for line in uncleaned_file.readlines())
    # add additional step to allow force mode
    cleaned_file.writelines((line for line in process_generator if line is not None))
  print("Cleaning completed, exported from {:s} to {:s}".format(file_in_dir, file_out_dir))
