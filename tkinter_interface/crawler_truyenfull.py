from crawler_wenku import get_from_url, whole_downloader
import io, re, json, os
import html2text
import argparse

truyenfull_base_link = "https://truyenfull.vn"
truyenfull_category_link = "https://truyenfull.vn/the-loai/{:s}/trang-{:d}/"
truyenfull_completed_category_link = "https://truyenfull.vn/the-loai/{:s}/hoan/trang-{:d}/"
truyenfull_book_link = "https://truyenfull.vn/{:s}"
truyenfull_list_chapter_link = "https://truyenfull.vn/{:s}/trang-{:d}"
truyenfull_chapter_link = "https://truyenfull.vn/{:s}/{:s}"

truyenfull_category_regex = re.compile("https:\/\/truyenfull\.vn\/the-loai\/(.+?)\/")
def category_searcher(original_link=truyenfull_base_link, category_link_regex=truyenfull_category_regex):
	link_data = get_from_url(original_link, stream=None)
	return set(re.findall(category_link_regex, link_data))

truyenfull_index_regex = re.compile("https:\/\/truyenfull.vn\/the-loai\/.+?\/trang-(\d+?)\/")
truyenfull_book_regex = re.compile("<a href=\"https:\/\/truyenfull.vn\/([\w\-]+?)\/\" title=\"(.+?)\" itemprop=\"url\">")
def book_searcher(category_link, search_link=None, index_link_regex=truyenfull_index_regex, book_link_regex=truyenfull_book_regex):
	search_link = search_link or category_link(1)
	search_data = get_from_url(search_link, stream=None)
	indices = re.findall(index_link_regex, search_data)
	max_index = max([int(idx) for idx in indices]) if len(indices) > 0 else 1
	books = set()
	for idx in range(1, max_index+1):
		search_data = get_from_url(category_link(idx), stream=None)
		books_data = re.findall(book_link_regex, search_data)
		books.update([book_raw for book_raw, book_name in books_data])
	return books

truyenfull_chapter_regex = re.compile("https:\/\/truyenfull.vn\/([\w\-]+?)\/([\w\d\-]+?)\/")
truyenfull_chapter_list_regex = re.compile("https:\/\/truyenfull.vn\/([\w\-]+?)\/trang\-(\d+?)\/")
def chapter_searcher(book_name, book_link=truyenfull_book_link, chapter_list_link=truyenfull_list_chapter_link, chapter_list_regex=truyenfull_chapter_list_regex, chapter_regex=truyenfull_chapter_regex):
	search_link = book_link.format(book_name)
	search_data = get_from_url(search_link, stream=None)
	indices_with_names = re.findall(chapter_list_regex, search_data)
	max_index = max([int(idx) for name, idx in indices_with_names if name == book_name]) if len(indices_with_names) > 0 else 1
	chapters = set()
	for idx in range(1, max_index+1):
		search_data = get_from_url(chapter_list_link.format(book_name, idx), stream=None)
		chapters_data = re.findall(chapter_regex, search_data)
		chapters.update([chapter_name for name, chapter_name in chapters_data if name == book_name and "trang" not in chapter_name])
	return list(chapters)

def _chapter_handler(book_data):
	book_name, book_total_chapters = book_data
	# take book_name, return the locations
	return book_name, (("{:s}.txt".format(chapter_name), truyenfull_chapter_link.format(book_name, chapter_name)) for chapter_name in book_total_chapters)

def _create_config():
	config = {}
	categories = config["categories"] = list(category_searcher())
	books = set()
	completed_books = set()

	for category in categories:
		# all
		category_link_fn = lambda idx: truyenfull_category_link.format(category, idx)
		list_books_found = book_searcher(category_link_fn)
		books.update(list_books_found)
		# completed
		category_link_fn = lambda idx: truyenfull_completed_category_link.format(category, idx)
		completed_books_found = book_searcher(category_link_fn)
		completed_books.update(completed_books_found)
		print("Book category {:s} done, found books {:d}({:d})".format(category, len(list_books_found), len(completed_books_found)))
	config["books"] = list(books)
	config["completed"] = list(completed_books)
	return config

def _load_config(config_file):
	with io.open(config_file, "r", encoding="utf-8") as config_file:
		config = json.load(config_file)
	return config

def _save_config(config, config_file):
	with io.open(config_file, "w", encoding="utf-8") as config_file:
		json.dump(config, config_file)
	return config

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="truyenfull.vn page crawler")
	parser.add_argument('-s', "--save_path", type=str, default="crawler_data/truyenfull", help="Directory to save crawling data into")
	parser.add_argument('-c', "--config_file", type=str, default="crawler.json", help="Overall data from previous run")
	parser.add_argument('-m', "--mode", type=str, default="all", choices=["all", "completed", "length", "selected"], help="Mode to choose to download books to. Default all")
	parser.add_argument("--length_limit", type=int, default=1000, help="If using length, only download those with more than this limit")
	parser.add_argument("--reload_books", action="store_true", help="If specified, reload all books data")
	parser.add_argument("--reload_chapters", action="store_true", help="If specified, reload all chapters data")
	parser.add_argument("--selected_series", type=str, nargs="*", default=None, help="In selected mode, load only the books specified by this arguments")
	parser.add_argument("--debug", action="store_true", help="If specified, run specific debug action")
	args = parser.parse_args()
	download_save_path = os.path.join( os.path.dirname(os.path.realpath(__file__)), args.save_path)
	
	print("Checking library...")
	if not os.path.isdir(args.save_path):
		os.makedirs(args.save_path)
	config_file = os.path.join(download_save_path, args.config_file)
	if(os.path.isfile(config_file) and not args.reload_books):
		# TODO reload books only affect books/completed, keeping chapters/downloaded
		config = _load_config(config_file)
		backup_config_file = os.path.join(download_save_path, "backup.json")
		import shutil
		shutil.copyfile(config_file, backup_config_file)
	else:
		config = _save_config(_create_config(), config_file)
	
	print("Checking book chapters..")
	books = config["books"] if args.mode != "completed" else config["completed"]
	chapters = config["chapters"] = config.get("chapters", dict())
	valid_books = []
	# get a list of valid books
	for book_name in books:
		if(book_name not in chapters or args.reload_chapters):
			chapters[book_name] = chapter_searcher(book_name)
			config = _save_config(config, config_file)
			print("Re-querrying chapters of book {:s} done, chapter count {:d}".format( book_name, len(chapters[book_name]) ))
		chapter_count = len(chapters[book_name])
		if(args.mode == "selected" and book_name in args.selected_series):
			valid_books.append(book_name)
		elif(args.mode == "length" and chapter_count > args.length_limit):
			valid_books.append(book_name)
		if(args.debug):
			chapters[book_name] = [item for item in chapters[book_name] if "trang" not in item]
			config = _save_config(config, config_file)
			print("Done reloading for book {:s}".format(book_name))
	# download using the list of valid_books
	
	print("Initiate download..")
	download_data = ((book_name, chapters[book_name]) for book_name in valid_books)
	text_parser = html2text.HTML2Text()
	whole_downloader(download_data, _chapter_handler, save_location=download_save_path, parser=text_parser)
