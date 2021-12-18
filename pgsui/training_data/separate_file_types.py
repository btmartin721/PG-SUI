import os
import sys
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

def main():
	print("File Counts:\n")

	path = "training_data/extracted/"
	extension_list = [
		".phy", 
		".PHY",
		".phylip",
		".PHYLIP", 
		".str", 
		".STR",
		".ustr",
		".USTR", 
		".structure",
		".STRUCTURE", 
		".loci",
		".LOCI", 
		".vcf",
		".VCF",
		".Vcf", 
		".nex",
		".NEX", 
		".nexus",
		".NEXUS"
	]

	totals, overall_totals = count_filetypes(extension_list, path)

	for k, v in totals.items():
		print("{}: {}".format(k, v))

	print("Total files: {}".format(overall_totals))

	# Subset CSV file to only remaining files.
	df = pd.read_csv("training_data/metadata/metadata_genomic_data.csv")
	ext_dirs = os.listdir(path)

	df2 = df[df["DatasetID"].isin(ext_dirs)]

	df2.to_csv("training_data/metadata/metadata_genomic_final_sangerRemoved.csv", header=True, index=False)

	print("Datasets remaining: {}".format(len(df2)))

	outpath = "training_data/collated"

	extension_list2 = [
		"phy", 
		"str", 
		"structure",
		"loci",
		"vcf",
		"nex",
	]

	for ext in extension_list2:
		if ext == "phy":
			copy_files(ext, outpath, path, alt_ext="phylip")
		elif ext == "str":
			copy_files(ext, outpath, path, alt_ext="structure")
		elif ext == "nex":
			copy_files(ext, outpath, path, alt_ext="nexus")
		else:
			copy_files(ext, outpath, path, alt_ext=None)
		
def count_filetypes(extension_list, mypath):
	totals = dict()
	for ext in extension_list:
		file_count = sum(f.endswith(ext) for _, _, files in os.walk(mypath) for f in files)
		totals[ext] = file_count
	
	# Get overall totals among all file types
	overall_totals = 0
	for k, v in totals.items():
		overall_totals += v
	
	return totals, overall_totals

def copy_files(ext, dest, srcdir, alt_ext=None):

	outdir = "{}/{}".format(dest, ext)
	Path(outdir).mkdir(parents=True, exist_ok=True)

	for resdir in os.listdir(srcdir):
		for root, dirs, files in os.walk(os.path.join(srcdir, resdir)):
			for myfile in files:
				if alt_ext is not None:
					if myfile.lower().endswith(ext) or \
					myfile.lower().endswith(alt_ext):
						shutil.copy(os.path.join(root, myfile), os.path.join(outdir, "{}_{}".format(resdir, myfile)))
				else:
					if myfile.lower().endswith(ext):
						shutil.copy(os.path.join(root, myfile), os.path.join(outdir, "{}_{}".format(resdir, myfile)))

if __name__ == "__main__":
	main()

