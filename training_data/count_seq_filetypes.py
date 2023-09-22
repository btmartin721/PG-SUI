import os
import pandas as pd
import numpy as np
import sys
import shutil
import glob

def main():

	df = pd.read_csv("training_data/metadata/metadata_genomic_data.csv")

	path = "training_data/extracted/"

	for entry in os.scandir(path):
		if os.path.isdir(entry.path) and not os.listdir(entry.path):
			os.rmdir(entry.path)

	notempty_resdirs = os.listdir(path)

	df = df[df["DatasetID"].isin(notempty_resdirs)]
	genomic_files = df["DatasetID"].values.tolist()

	ext_dirs = os.listdir(path)
	print("{} results non-empty directories remaining".format(len(ext_dirs)))

	counter = 0
	for d in ext_dirs:
		if d not in genomic_files:
			shutil.rmtree("{}/{}".format(path, d))

	for dirs in os.listdir("{}".format(path)):
		my_dir = "{}/{}".format(path, dirs)
		for fname in os.listdir(my_dir):
			if fname.startswith("_"):
				shutil.rmtree(os.path.join(my_dir, fname))

	final_dirs = os.listdir(path)

	print("Remaining Genomic Directories: {}".format(len(final_dirs)))

	extension_list = [".phy", ".phylip", ".str", ".ustr", ".structure", ".loci", ".vcf", ".nex", ".nexus"]

	# Count retained files.
	totals = dict()
	for ext in extension_list:
		file_count = sum(f.endswith(ext) for _, _, files in os.walk(path) for f in files)
		totals[ext] = file_count

	print("File Counts\n")
	overall_totals = 0
	for k, v in totals.items():
		print("{}: {}".format(k, v))
		overall_totals += v
	print("Total number of files: {}".format(overall_totals))

if __name__ == "__main__":
	main()