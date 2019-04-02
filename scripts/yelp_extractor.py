import io, json, sys

if __name__ == "__main__":  
  input_path, output_path = sys.argv[1:3]
  print("{} -> {}".format(input_path, output_path))
  with io.open(input_path, "r") as input_file, io.open(output_path, "w") as output_file:
    yelp_data = ( json.loads(line) for line in input_file.readlines())
    # convert and dump
    lines_and_ratings = [ (rec["text"], rec["stars"]) for rec in yelp_data ]
    del yelp_data
    lines, ratings = [list(arr) for arr in  zip(*lines_and_ratings)]
    json.dump( [{"lines": lines, "ratings": ratings}], output_file)
