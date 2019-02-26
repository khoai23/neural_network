import sys, io, re
import html

NUMBER_RANGE = 3
DEFAULT_SENTENCE_LENGTH = 50
COUNTER_PRINT = 1000

# refer to https://stackoverflow.com/questions/37579692/unicode-range-for-vietnamese
# and http://kipalog.1upnote.me/post/KTfPN3Gkv43ykRQ1aRd5_w

# all_vietnamese_characters_no_tone_marks = u"aAăĂâÂbBcCdDđĐeEêÊfFgGhHiIjJkKlLmMnNoOôÔơƠpPqQrRsStTuUưƯvVwWxXyYzZ"
all_vietnamese_characters = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
all_vowels_no_diacritics = "AaEeIiOoUu"
all_diacritics_vowels_no_tone_marks="ÂâĂăƠơƯưÊê"
all_vowels_no_tone_marks = all_vowels_no_diacritics + all_diacritics_vowels_no_tone_marks
all_vowels_tone_marks = "àÀảẢãÃáÁạẠằẰẳẲẵẴắẮặẶầẦẩẨẫẪấẤậẬđĐèÈẻẺẽẼéÉẹẸềỀểỂễỄếẾệỆìÌỉỈĩĨíÍịỊòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘờỜởỞỡỠớỚợỢùÙủỦũŨúÚụỤừỪửỬữỮứỨựỰỳỲỷỶỹỸýÝỵỴ"

# dictionary
diacritic_default = all_vowels_no_diacritics
def generate_tranform_dict():
  all_ghost_diacritics = "\u0300\u0301\u0302\u0303\u0309\u0323\u02C6\u0306\u031B"
  # dia
  add_circumflex = "AaEeOoÁÀÃẠẢáàãạảÉÈẼẸẺéèẽẹẻÓÒÕỌỎóòõọỏ", "ÂâÊêÔôẤẦẪẬẨấầẫậẩẾỀỄỆỂếềễệểỐỒỖỘỔốồỗộổ", "\u02C6"
  add_alternative_circumflex = "AaEeOoÁÀÃẠẢáàãạảÉÈẼẸẺéèẽẹẻÓÒÕỌỎóòõọỏ", "ÂâÊêÔôẤẦẪẬẨấầẫậẩẾỀỄỆỂếềễệểỐỒỖỘỔốồỗộổ", "\u0302"
  add_breve = "AaÁÀÃẠẢáàãạả", "ĂăẮẰẴẶẲắằẵặẳ", "\u0306"
  add_horn = "OoUuÓÒÕỌỎóòõọỏ", "ƠơƯưỚỜỠỢỞớờỡợở", "\u031B"
  # tone_marks
  add_acute = "AaEeIiOoUuÂâĂăƠơƯưÊêÔôy", "ÁáÉéÍíÓóÚúẤấẮắỚớỨứẾếỐốý", "\u0301"
  add_grave = "AaEeIiOoUuÂâĂăƠơƯưÊêÔôy","ÀàÈèÌìÒòÙùẦầẰằỜờỪừỀềỒồỳ", "\u0300"
  add_tilde = "AaEeIiOoUuÂâĂăƠơƯưÊêÔôy","ÃãẼẽĨĩÕõŨũẪẫẴẵỠỡỮữỄễỖỗỹ", "\u0303"
  add_under_dot = "AaEeIiOoUuÂâĂăƠơƯưÊêÔôy","ẠạẸẹỊịỌọỤụẬậẶặỢợỰựỆệỘộỵ", "\u0323"
  add_above_hook = "AaEeIiOoUuÂâĂăƠơƯưÊêÔôy","ẢảẺẻỈỉỎỏỦủẨẩẲẳỞởỬửỂểỔổỷ", "\u0309"
#  print(locals())
  convert_dict = {}
  for name, item in locals().items():
    if(not isinstance(item, tuple) or len(item) != 3):
      continue
    # use the bunch above to perform conversion
    before_conv, after_conv, char = item
    convert_dict[char] = {k:v for k, v in zip(before_conv, after_conv)}
  print("All diacritics ordinal number: {}".format([ord(k) for k in convert_dict.keys()]))
  return all_ghost_diacritics, convert_dict

diacritics_swapper_regex = re.compile("([\u0301\u0300\u0303\u0323\u0309])([\u02C6\u0306\u0302\u031B])")
def swap_diacritics(line):
	# this swap the diacritics to the front before the tone marks
	return re.sub(diacritics_swapper_regex, "\2\1", line)

def vietnamese_ghost_characters_cleaner(line, ghost_checker, convert_dict, ghost_checker_regex=None, debug_line_number=None, ignore_error=False):
  # This is gonna be ugly...
  if ghost_checker_regex is None:
    ghost_checker_regex = re.compile("[{:s}]".format(ghost_checker))
  if re.findall(ghost_checker_regex, line):
    if(debug_line_number):
      print("Found ghost diacritics, line number {}".format(debug_line_number)) 
    new_line = ""
    for idx, char in enumerate(line):
      if char in ghost_checker:
#        print("Char [ {:s}]({}-{:d}) is ghost".format(char, char.encode(), idx))
        # some article have space before the diacritics, which should be removed 
        if(len(new_line) > 0 and new_line[-1] == ' '):
          print("Caught space for diacritic  {:s}, removing.".format(char))
          new_line = new_line[:-1]
        prev_char = new_line[-1] if idx > 0 else ""
        prev_char_changed = convert_dict[char].get(prev_char, None)
        # if prev_char cannot be gotten (invalid), depend on ignore error do it keep going/raise exception. Else, change the prev_char with the correct version (prev_char_changed)
        if(prev_char_changed is None):
          error_message = "Warning: cannot find replacement for {:s}(u{}) with diacritic {:s}(u{}).".format(prev_char, ord(prev_char), char, ord(char))
          if(ignore_error):
            print("Error suppressed @vietnamese_ghost_characters_cleaner: {:s}, Context: [{:s}]".format(error_message, line[max(idx-5, 0):min(idx+5, len(line))]))
          else:
            raise KeyError(error_message)
        else:
          new_line = new_line[:-1] + prev_char_changed
      else:
        new_line += char
    return new_line
  return line

# comment @ https://stackoverflow.com/questions/1366068/whats-the-complete-range-for-chinese-characters-in-unicode
chinese_char_regex = re.compile("[\u2E80-\u2FD5\u3190-\u319f\u3400-\u4DBF\u4E00-\u9FCC]")
def chinese_character_identifier(char):
	return re.match(chinese_char_regex, char)

def split_line_chinese(line):
	new_line = []
	past_other_chars = None
	for char in line:
		if(chinese_character_identifier(char)):
			if(past_other_chars):
				new_line.append(past_other_chars)
				past_other_chars = None
			new_line.append(char)
		else:
			if(char == " "):
				if(past_other_chars):
					new_line.append(past_other_chars)
					past_other_chars = None
				pass
			elif(past_other_chars):
				past_other_chars += char
			else:
				past_other_chars = char
	if(past_other_chars):
		new_line.append(past_other_chars)
	return new_line

# this to remove html entities
# taken directly from http://effbot.org/zone/re-sub.htm#unescape-html
# modified slightly
html_entity_regex = re.compile("&#?\w+;")
def html_entity_fixup(m):
  text = m.group(0)
  if text[:2] == "&#":
    # character reference
    try:
      if text[:3] == "&#x":
        return chr(int(text[3:-1], 16))
      else:
        return chr(int(text[2:-1]))
    except ValueError:
      print("Not found value for text {:s}".format(text))
      pass
  else:
    # named entity
    try:
      text = chr(html.entities.name2codepoint[text[1:-1]])
    except KeyError:
      print("Not found value for text {:s}".format(text[1:-1]))
      pass
  return text # leave as is
def replace_html_entity_reg(line):
  return re.sub(html_entity_regex, html_entity_fixup, line)

def replace_html_entity(line):
  return html.unescape(line)

html_tag_regex = re.compile("<.*?>")
# brute force removal
def remove_html_tags(line):
  return re.sub(html_tag_regex, " ", line)

remove_leading_idx = re.compile("\\d+\t")
# wikipedia have first number and tab
def remove_first_number(line):
  return re.sub(remove_leading_idx, "", line)

# images tokens are either {number}px or {number}x{number}px
image_token_regex = re.compile("\d+(x\d+)?px")
def remove_image_tokens(line):
  return re.sub(image_token_regex, "", line)

# model must contain one uppercase char/one number each
# see https://stackoverflow.com/questions/7684815/regex-pattern-to-match-at-least-1-number-and-1-character-in-a-string for the explaination of lookahead
model_regex = re.compile("(?=[A-Z]+\-*[0-9]|[0-9]+\-*[A-Z])[A-Z0-9\-]{2,}(?=\s|$)")
def replace_model_token(line):
#  raise Exception("Faulty regex")
  return re.sub(model_regex, "<model> ", line)

# emoticon default splitter
emoticon_regex = re.compile("([{:s}-{:s}])".format(chr(0x1f600), chr(0x1f64f)))
def split_emoticon(line):
	return re.sub(emoticon_regex, " \1 ", line)

# rejoin character plus . (eg: S. Truman)
middle_name_regex = re.compile("(?=^|\s) ([A-Z])\s\.")
def rejoin_middle_name(line):
  return re.sub(middle_name_regex, " \1.", line)

# replace number with larger than NUMBER_RANGE amount of digits
number_replace_regex = re.compile("(?=^|\s)(\s*)\d{" + str(NUMBER_RANGE) + ",}")
# replace detected float (type: 30,52 or 14.12)
float_number_replace_regex = re.compile("\d+[\.\,]\d+")
float_number_replace_after_tokenization_regex = re.compile("\d+ [\.\,] \d+")
def replace_number_tokens(line, tokenized=False):
  float_replace_regex = float_number_replace_regex if(not tokenized) else float_number_replace_after_tokenization_regex
  line = re.sub(float_replace_regex, "<float>", line)
  line = re.sub(number_replace_regex, " <number>", line)
  return line

# milestones in wikipedia seem to have this format
# match ({single_char}.
# this is only compatible before cleaning
milestone_regex = re.compile("\d{3,4} \) \* \d{3,4} \- (.+?) \([^\d\W]\.")
milestone_regex_after_clean = re.compile("\d{3,4} \) \* \d{3,4} \- (.+?) \( [^\d\W] \.")
def remove_milestone(line, tokenized=False):
  milestone_remove_regex = milestone_regex if(not tokenized) else milestone_regex_after_clean
  return re.sub(milestone_remove_regex, "\1", line)

def replace_unrecognized_characters(line):
  raise NotImplementedError("This requires a list of valid characters to work")

cap_token = "<cap>"
all_cap_token = "<caps>"
def decapitalize(word, consider_all_caps=False):
  if(word[0].isupper() and (not word.isupper() or len(word) == 1)):
    return [cap_token, word.lower()]
  elif(consider_all_caps and word.isupper()):
    return [all_cap_token, word.lower()]
  else:
    return [word]

def add_capitalization(tokens, consider_all_caps=False):
  return [ token for orig_token in tokens for token in decapitalize(orig_token, consider_all_caps) ]

def splice_long_sentences(tokens, threshold=50, overlap=10):
  # if number of tokens larger than threshold, splice it into separate sets of threshold-length tokens
  if(len(tokens) <= threshold):
    return [tokens]
  else:
    # splicing basing on the overlap
    num_split = (len(tokens) - threshold) // overlap + 1
    return [tokens[starter*10:starter*10+threshold] for starter in range(num_split)]

REFORMATTER = ( ("-RRB", ")"), ("-LRB-", ")"), ("``", "\""), ("\'\'", "\""))
def misc_reformat(line):
	for src, tgt in REFORMATTER:
		line = line.replace(src, tgt)
	return line

if __name__ == "__main__":
  mode = sys.argv[1]
  file_in_dir = sys.argv[2]
  file_out_dir = sys.argv[3]
  # Do everything.
  with io.open(file_in_dir, "r", encoding="utf8") as uncleaned_file, io.open(file_out_dir, "w", encoding="utf8") as cleaned_file:
  #  lines = uncleaned_file.readlines()
    global counter
    counter = 0
    def process_line(orig_line):
      # add boredom-resistant counter
      global counter
      counter += 1
      if(counter % COUNTER_PRINT == 0):
        print("Lines processed: {:d}".format(counter))
      # process
      line = orig_line.strip()
      if(not line):
        # blank, output to stdout and return it without \n
        print("Empty line detected @{:d}".format(counter))
        return ""
      try:
        if("wiki" in mode):
          line = remove_first_number(line)
          line = remove_image_tokens(line)
          line = remove_milestone(line)
          line = replace_number_tokens(line)
        tokens = line.split()
        if("cap" in mode):
          tokens = add_capitalization(tokens, consider_all_caps=True)
      except Exception as e:
        print("Exception caught @ line {:d}-{:s}".format(counter, orig_line))
        raise e
      return " ".join(tokens) + "\n"
    cleaned_file.writelines((process_line(line) for line in uncleaned_file.readlines()))
  print("Cleaning completed, exported from {:s} to {:s}".format(file_in_dir, file_out_dir))
