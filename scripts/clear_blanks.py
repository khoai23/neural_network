import io
train_location_en = "/home/quan/Workspace/Data/iwslt15/train.en"
train_location_vi = "/home/quan/Workspace/Data/iwslt15/train.vi"

train_location_en_t = "/home/quan/Workspace/Data/iwslt15/train.filter.en"
train_location_vi_t = "/home/quan/Workspace/Data/iwslt15/train.filter.vi"
with io.open(train_location_en, "r", encoding="utf-8") as en_file, \
		io.open(train_location_vi, "r", encoding="utf-8") as vi_file:
	en_data = en_file.readlines()
	vi_data = vi_file.readlines()

filtered_data = list(zip(en_data, vi_data))
print(len(filtered_data))
vi_data, en_data = zip( *[tup for tup in filtered_data if tup[0].strip() != "" and tup[1].strip() != ""])
with io.open(train_location_en_t, "w", encoding="utf-8") as en_file, \
		io.open(train_location_vi_t, "w", encoding="utf-8") as vi_file:
	en_file.write("\n".join(en_data))
	vi_file.write("\n".join(vi_data))
