from subprocess import call
import io
recover_hashes_file = "_recover_hashes.txt"
# run the recover
recover_command = ["git", "fsck", "--cache", "--unreachable"]
with io.open(recover_hashes_file, "w", encoding="utf-8") as hashes_file:
	call(recover_command, stdout=hashes_file)
filename = "recover_{:d}"
with io.open(recover_hashes_file, "r", encoding="utf-8") as commits_file:
	commits = commits_file.readlines()
	for i, commit_obj in enumerate(commits):
		commit_hash = commit_obj.replace("unreachable blob ", "").strip()
		with io.open(filename.format(i), "wb") as recover_file:
			call(["git", "show", commit_hash], stdout=recover_file)
