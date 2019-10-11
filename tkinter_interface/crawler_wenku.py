import io, os, time, sys, re
import urllib.request as request
from urllib.error import HTTPError
import chardet
import html2text
import argparse
# from html.parser import HTMLParser
# import requests

wenku_directory_book = "https://www.wenku8.net/book/{}.htm"
wenku_directory = "https://www.wenku8.net/novel/{}/{}/{}"

string_invalid_book = "因版权问题，文库不再提供该小说的在线阅读与下载服务！"
def check_is_invalid(content):
	return content.find(string_invalid_book) >= 0

string_incomplete_book = "文章状态：连载中"
string_complete_book = "文章状态：已完成"
def check_book_is_incomplete(content, strict=True):
	found_incomplete = content.find(string_incomplete_book) >= 0
	found_complete = content.find(string_complete_book) >= 0
	if(strict):
		assert found_incomplete != found_complete, "Error with completion checker: complete {} == incomplete {}".format(found_complete, found_incomplete)
	return found_incomplete

def check_is_int(val):
	try:
		val = int(val)
	except ValueError:
		return False
	return val >= 0

wenku_table_regex = re.compile("<td class=\"ccss\"><a href=\"(.+)\">.+?</a></td>")
def wenku_catch_links(content):
	return re.findall(wenku_table_regex, content)

def get_from_url(url, stream=sys.stdout, content_fn=None, parser=None):
	request_response = request.urlopen(url)
	content = request_response.read()
	decode_charset = request_response.headers.get_content_charset() or chardet.detect(content)["encoding"] or "utf8"
	print("Open url {}, charset {}".format(url, decode_charset))
	content = content.decode(decode_charset, errors='ignore')
	request_response.close()
	if(parser):
		stream and stream.write(parser.handle(content))
	else:
		stream and stream.write(content)
	if(content_fn):
		content_result = content_fn(content)
		return content_result
	elif(stream is None):
		return content

def search_book_range(list_crawler_target, additional_valid_func=None, text_parser=None):
	# regardless of which mode, search through the range for chapters
	# the additional_constraint_function must output True, (smth) or False, error_message
	download_data = []
	assert text_parser is not None, "Text parser required!"
	if(additional_valid_func):
		print("Merging additional validity function")
		book_check_valid_function = lambda x: check_is_invalid(x) and additional_valid_func(x)
	else:
		book_check_valid_function = check_is_invalid
		
	start_timer = time.time()
	for idx in list_crawler_target:
		sub_idx = int(idx) // 1000
		wenku_main_url = wenku_directory.format(sub_idx, idx, "index.htm")
		try:
			list_chapters = get_from_url(wenku_main_url, stream=None, content_fn=wenku_catch_links)
		except HTTPError:
			# book with idx not found
			print("HTTPError for url {}, ignoring.".format(wenku_main_url))
			list_chapters = []
		# check if the link is valid
		if(not get_from_url(wenku_directory_book.format(idx), stream=None, parser=text_parser, content_fn=book_check_valid_function)):
			download_data.append( (len(list_chapters), idx, list_chapters) )
		else:
			print("Book {} deemed invalid, purging from the set".format(idx))
	print("Completed search for the book range, {:d} valid items, time passed {:.2f}".format(len(download_data), time.time()-start_timer))
	return download_data

def whole_downloader(download_data, sub_item_function, save_location="./", content_fn=None, parser=None, print_debug=False, book_end_hook_fn=None):
	# download_data is an iterator of downloadable data
	# sub_item_function take the data and return the folder and the iterable inside that folder
	# e.g book idx and the list of item in that index
	# link function take the data and return the corresponding download link
	for data in download_data:
		start_timer = time.time()
		folder_name, items = sub_item_function(data)
		folder_path = os.path.join(save_location, folder_name)
		if(not os.path.isdir(folder_path)):
			print("Folder @ {:s} unavailable, creating new..".format(folder_path))
			os.mkdir(folder_path)
		# the second is an iterable which return a tuple of item name and location
		for item_name, item_data_link in items:
			# create a file with name
			item_path = os.path.join(folder_path, item_name)
			if(os.path.isfile(item_path)):
				print("file at {:s} detected, skipping.".format(item_path))
				continue
			with io.open(item_path, "w", encoding="utf-8") as item_file:
				get_from_url(item_data_link, stream=item_file, content_fn=content_fn, parser=parser)
		if(print_debug):
			print("Finished for item with folder name {:s} at path {:s}, costed {:.2f}s".format(folder_name, folder_path, time.time() - start_timer))
		if(book_end_hook_fn):
			book_end_hook_fn(folder_name)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Wenku page crawler")
	parser.add_argument("range", type=int, nargs='*', help="Range to load data into")
	parser.add_argument('-s', "--save_path", type=str, default="crawler_data/wenku", help="Directory to save crawling data into")
	parser.add_argument('-m', "--mode", type=str, choices=["range", "search", "completed", "default"], default="default", help="Mode of searching books.")
	args = parser.parse_args()
	
	executor_file_path = os.path.dirname(os.path.realpath(__file__))
	save_directory_orig = os.path.join(executor_file_path, args.save_path)
	if not os.path.isdir(save_directory_orig):
		os.makedirs(save_directory_orig)
	text_parser = html2text.HTML2Text()
	
	start = time.time()
	possible_mode_string = args.mode
	if(possible_mode_string == "range" or possible_mode_string == "search" or possible_mode_string == "completed"):
		start_range, end_range = args.range[:2]
		list_crawler_target = [str(idx) for idx in range(start_range, end_range)]
	else:
		list_crawler_target = [idx for idx in args.range if check_is_int(idx)]

	if(possible_mode_string == "search"):
		download_data = search_book_range(list_crawler_target, text_parser=text_parser)
		# add a search mode which will only download the 5% largest title basing on their count
		print("Initiating search and get the bigshots...")
		sort_size = (end_range - start_range) // 20
		kept_books = sorted(download_data, reverse=True)[:sort_size]
		print("Categorized by chapter number; best ({}-{} chaps); last ({}-{} chaps), downloading {} books".format(kept_books[0][1], kept_books[0][0], kept_books[-1][1], kept_books[-1][0], sort_size))
		download_data = kept_books
	elif(possible_mode_string == "completed"):
		# only take completed book
		print("Only take completed book from the range...")
		download_data == search_book_range(list_crawler_target, additional_valid_func=check_book_is_incomplete, text_parser=text_parser)
	else:
		# download everything in the range
		print("Download the whole range: start {} to end {}".format(start_range, end_range))
		download_data = search_book_range(list_crawler_target, text_parser=text_parser)

	# start the whole downloader
	# each data is a tuple of chapter_nums, book_idx, [chapter_sub_page]
	def data_to_name_and_pages(data):
		_, book_idx, list_chapters = data
		sub_idx = int(book_idx) // 1000
		return book_idx, (("{}_{}.txt".format(book_idx, idx), wenku_directory.format(sub_idx, book_idx, chapter_page)) for idx, chapter_page in enumerate(list_chapters))
	print("Initiate downloading the contents..")
	whole_downloader(download_data, data_to_name_and_pages, save_location=save_directory_orig, parser=text_parser, print_debug=True)
