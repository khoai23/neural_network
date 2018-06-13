from ciseau import tokenize, sent_tokenize
import io, sys

CONVERT_VI = False
CONVERT_CH = True
def condenseSentences(obj):
	if(isinstance(obj[0], str)):
		return obj
	elif(isinstance(obj[0], list)):
		return [item for sublist in obj for item in sublist]
	else:
		raise ValueError
if(CONVERT_VI):
	fileIn = io.open('vi_untokenized.txt', 'r', encoding='utf-8')
	data = fileIn.readlines()
	data = [condenseSentences(sent_tokenize(line, keep_whitespace=False)) for line in data]
	data = [' '.join(words) for words in data]
	fileOut = io.open('vi_tokenized.txt', 'w', encoding='utf-8')
	for line in data:
		fileOut.write(line.strip() + '\n')
	fileIn.close()
	fileOut.close()

if(CONVERT_CH):
	fileIn = io.open('ch_untokenized.txt', 'r', encoding='utf-8')
	data = fileIn.readlines()
	data = [' '.join(chars) for chars in data]
	fileOut = io.open('ch_tokenized.txt', 'w', encoding='utf-8')
	for line in data:
		fileOut.write(line.strip() + '\n')
	fileIn.close()
	fileOut.close()

