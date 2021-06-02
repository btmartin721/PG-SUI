
import zipfile
import os
from pathlib import Path

def get_datafiles():

	extension_list = [".phy", ".phylip", ".str", ".structure", ".loci", ".vcf", ".nex", ".nexus"]

	storage_path = "training_data/extracted"
	Path(storage_path).mkdir(parents=True, exist_ok=True)

	zipfiles = os.listdir("training_data/download/")

	num_zipfiles = len(zipfiles)

	for i, myzip in enumerate(zipfiles, start=1):
		outdir = "{}/results{}".format(storage_path, i)
		Path(outdir).mkdir(parents=True, exist_ok=True)
		with zipfile.ZipFile(myzip, "r") as zip_ref:
			for filename in zip_ref.namelist():
				seqfiles = list(filter(filename.endswith, extension_list))
				if seqfiles:
					for seqfile in seqfiles:
						zip_ref.extract(seqfile, outdir)



get_datafiles()