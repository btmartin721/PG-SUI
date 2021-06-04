
import zipfile
import os
import re
import io
import sys
from pathlib import Path
from tqdm import tqdm


def unzip_nested_zip_seqfiles(zip_ref, storage_path, extension_list, i, outdir, counter):
	"""[Recursively unzips zip files and extracts only files with a list of specific extensions as designated by extension_list]

	Args:
		zip_ref ([zipfile.Zipfile]): [Reference object opened with zipfile.Zipfile]

		storage_path ([str]): [Root path where the extracted files are to be saved when looping through a directory of zipfiles]

		extension_list ([list(str)]): [List of filename extensions to extract]

		i ([int]): [Current iteration, starting at 1, when looping through a directory of zipfiles]

		outdir ([str]): [Directory where the extracted files are to be saved]

		counter ([int]): [Count of the number of files extracted]

	Returns:
		[int]: [Count of the total number of files extracted]
	"""
	for filename in zip_ref.namelist():
		if any(filename.endswith(ext) for ext in extension_list):
			counter += 1
			try:
				zip_ref.extract(filename, outdir)

			except FileNotFoundError:
				continue

		if filename.endswith(".zip"):
			zfiledata = io.BytesIO(zip_ref.read(filename))
			with zipfile.ZipFile(zfiledata) as zip_ref2:
				counter = unzip_nested_zip_seqfiles(zip_ref2, storage_path, extension_list, i, outdir, counter)
	return counter

def atoi(text):
	"""[Converts string to an integer]

	Args:
		text ([str]): [text with mixed string and integer characters]

	Returns:
		[int or str]: [If int, returns int; else returns string]
	"""
	if text.isdigit():
		return int(text)
	else:
		return text

def natural_keys(text):
	"""[Key to use in sort function. E.g. alist.sort(key=natural_keys). Does natural sorting in human order. E.g. mystr1, mystr2, mystr3]

	Original code from:
		http://nedbatchelder.com/blog/200712/human_sorting.html
			(See Toothy's implementation in the comments)

	Code adapted from:

		https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
	"""
	return [atoi(c) for c in re.split(r"(\d+)", text)]

extension_list = [".phy", ".PHY", ".phylip", ".PHYLIP", ".str", ".STR", ".ustr", ".USTR", ".structure", ".STRUCTURE", ".loci", ".LOCI", ".vcf", ".VCF", ".Vcf", ".nex", ".NEX", ".nexus", ".NEXUS"]

storage_path = "training_data/extracted"
Path(storage_path).mkdir(parents=True, exist_ok=True)

# List all zipfiles and sort them in natural order (i.e. human order)
# So e.g. results1, results2, results3, etc.
zipfiles = os.listdir("training_data/download/")
zipfiles.sort(key=natural_keys)
zipfiles = ["training_data/download/{}".format(z) for z in zipfiles]

counter = 0
for cnt, myzip in enumerate(tqdm(zipfiles, desc="Extracting Zipfiles: "), start=1):
	outdir = "{}/results{}".format(storage_path, cnt)
	Path(outdir).mkdir(parents=True, exist_ok=True)

	try:
		with zipfile.ZipFile(myzip, "r") as zip_ref:
			counter = unzip_nested_zip_seqfiles(zip_ref, storage_path, extension_list, cnt, outdir, counter)

	except zipfile.BadZipFile:
		continue

print("Extracted {} total files".format(counter))