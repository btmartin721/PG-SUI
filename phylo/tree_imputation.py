import sys
import os 
import numpy as np
import dendropy as dp


def main():
	q = q_from_iqtree("example_data/trees/test.iqtree")
	tree = dp.Tree.get(path="example_data/trees/test.tre", schema="newick")
	data = readPhylip("example_data/phylip_files/test.phy")
	print(data)


def q_from_file(fname, label=True):
	q = blank_q_matrix()
	
	if not label:
		print("Warning: Assuming the following nucleotide order: A, C, G, T")
	
	with open(fname, "r") as fin:
		header=True
		qlines=list()
		for line in fin:
			line=line.strip()
			if not line:
				continue
			if header:
				if label:
					order = line.split()
					header=False
				else:
					order = ['A', 'C', 'G', 'T']
				continue
			else:
				qlines.append(line.split())
	fin.close()
	
	for l in qlines:
		for index in range(0,4):
			q[l[0]][order[index]] = float(l[index+1])
	printQ(q)
	

def q_from_iqtree(iqfile):
	q = blank_q_matrix()
	qlines=list()
	with open(iqfile, "r") as fin:
		foundLine=False
		matlinecount=0
		for line in fin:
			line=line.strip()
			if not line:
				continue
			if  "Rate matrix Q" in line:
				foundLine=True
				continue 
			if foundLine:
				matlinecount+=1
				if matlinecount > 4:
					break
				stuff=line.split()
				qlines.append(stuff)
			else:
				continue
	fin.close()
	
	#population q matrix with values from iqtree file
	order=[l[0] for l in qlines]
	for l in qlines:
		for index in range(0,4):
			q[l[0]][order[index]] = float(l[index+1])
	printQ(q)

def printQ(q):
	print("Rate matrix Q:")
	print("\tA\tC\tG\tT\t")
	for nuc1 in ['A', 'C', 'G', 'T']:
		print(nuc1, end="\t")
		for nuc2 in ['A', 'C', 'G', 'T']:
			print(q[nuc1][nuc2], end="\t")
		print("")

def blank_q_matrix(default=0.0):
	q=dict()
	for nuc1 in ['A', 'C', 'G', 'T']:
		q[nuc1] = dict()
		for nuc2 in ['A', 'C', 'G', 'T']:
			q[nuc1][nuc2] = default
	return(q)

#Function to read a phylip file. Returns dict (key=sample) of lists (sequences divided by site)
def readPhylip(phy):
	if os.path.exists(phy):
		with open(phy, 'r') as fh:
			try:
				num=0
				ret = dict()
				for line in fh:
					line = line.strip()
					if not line:
						continue
					num += 1
					if num == 1:
						continue
					arr = line.split()
					ret[arr[0]] = list(arr[1])
				return(ret)
			except IOError:
				print("Could not read file ",phy)
				sys.exit(1)
			finally:
				fh.close()
	else:
		raise FileNotFoundError("File %s not found!"%phy)

def impute_phylo(tree, genotypes, Q, rates):
	"""[Imputes genotype values on a provided guide 
		tree, assumping maximum parsimony]
	"""

if __name__ == "__main__":
	main()