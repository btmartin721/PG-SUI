import sys
import os 
import numpy as np
import scipy.linalg
import pandas as pd
import toytree as tt


def main():
	q = q_from_iqtree("example_data/trees/test.iqtree")
	#tree = dp.Tree.get(path="example_data/trees/test.tre", schema="newick", preserve_underscores=True)
	tree = tt.tree("example_data/trees/test.tre", tree_format=0)
	data = readPhylip("example_data/phylip_files/test.phy")
	
	imputed = impute_phylo(tree, data, q)
	

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
	qdf = pd.DataFrame(q)
	return(qdf.T)
	

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
	qdf = pd.DataFrame(q)
	return(qdf.T)

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

def impute_phylo(tree, genotypes, Q, site_rates=None, exclude_N=False):
	"""[Imputes genotype values on a provided guide 
		tree, assumping maximum parsimony]
	
	Sketch:
		For each SNP:
			1) if site_rates, get site-transformated Q matrix
			
			2) Postorder traversal of tree to compute ancestral 
			state likelihoods for internal nodes (tips -> root)
				If exclude_N==True, then ignore N tips for this step
				
			3) Preorder traversal of tree to populate missing genotypes
			with the maximum likelihood state (root -> tips)
	"""
	node_lik = dict()
	snp_index = 0
	
	#for each SNP: 
	
	#LATER: Need to get site rates 
	rate = 1.0
	
	site_Q = Q.copy(deep=True)*rate

	for node in tree.treenode.traverse("postorder"):
		if node.is_leaf():
			continue
		if node.idx not in node_lik:
			node_lik[node.idx] = None
		for child in node.get_leaves():
			#get branch length to child
			#bl = child.edge.length
			#get transition probs 
			pt = transition_probs(site_Q, child.dist)
			if child.is_leaf():
				if child.name in genotypes:
					#get genotype 
					sum = None
					for allele in get_iupac_full(genotypes[child.name][snp_index]):
						if sum is None:
							sum = list(pt[allele])
						else:
							sum = [sum[i] + val for i, val in enumerate(list(pt[allele]))]
					if node_lik[node.idx] is None:
						node_lik[node.idx] = sum
					else:
						node_lik[node.idx] = [sum[i] * val for i, val in enumerate(node_lik[node.idx])]
				else:
					#raise error 
					sys.exit("Error: Taxon",child.name,"not found in genotypes")
			else:
				l = get_internal_lik(pt, node_lik[child.idx])
				if node_lik[node.idx] is None:
					node_lik[node.idx] = l 
				else:
					node_lik[node.idx] = [l[i] * val for i, val in enumerate(node_lik[node.idx])]

def get_internal_lik(pt, lik_arr):
	ret = list()
	for i, val in enumerate(lik_arr):
		ret.append(val*pt.iloc[:,i])
	return(ret)

def transition_probs(Q, t):
	ret = Q.copy(deep=True)
	m = Q.to_numpy()
	pt = scipy.linalg.expm(m*t)
	ret[:] = pt
	return(ret)

def get_iupac_full(char):
	char = char.upper()
	iupac = {
		"A"	: ["A"],
		"G"	: ["G"],
		"C"	: ["C"],
		"T"	: ["T"],
		"N"	: ["A", "C", "T", "G"],
		"-"	: ["A", "C", "T", "G"],
		"R"	: ["A","G"],
		"Y"	: ["C","T"],
		"S"	: ["G","C"],
		"W"	: ["A","T"],
		"K"	: ["G","T"],
		"M"	: ["A","C"],
		"B"	: ["C","G", "T"],
		"D"	: ["A","G", "T"],
		"H"	: ["A","C", "T"],
		"V"	: ["A","C", "G"]
	}
	ret = iupac[char]
	return(ret)

if __name__ == "__main__":
	main()