#!/usr/bin/python

import re
import sys
from itertools import product
from collections import Counter

def blacklist_missing(loci, threshold, iupac=False):
	blacklist=list()
	for i in range(0, len(loci)):
		alleles = expandLoci(loci[i], iupac=False)
		c = Counter(alleles)
		if float(c[-9]/sum(c.values())) > threshold:
			blacklist.append(i)
	return(blacklist)

def blacklist_maf(loci, threshold, iupac=False):
	blacklist=list()
	for i in range(0, len(loci)):
		alleles = expandLoci(loci[i], iupac=False)
		c = Counter(alleles)
		if len(c.keys()) <= 1:
			blacklist.append(i)
			continue
		else:
			minor_count = c.most_common(2)[1][1]
			if float(minor_count/sum(c.values())) < threshold:
				blacklist.append(i)
	return(blacklist)

#list of genotypes in 0-1-2 format
def expandLoci(loc, iupac=False):
	ret=list()
	for i in loc:
		if not iupac: 
			ret.extend(expand012(i))
		else:
			ret.extent(get_iupac_caseless(i))
	return(ret)

def expand012(geno):
	g=str(geno)
	if g == "0":
		return([0,0])
	elif g == "1":
		return([0,1])
	elif g == "2":
		return([1,1])
	else:
		return([-9,-9])
		

def remove_items(all_list, bad_list):
	# using list comprehension to perform the task
	res = [i for i in all_list if i not in bad_list]
	return res

def count_alleles(l, vcf=False):
	"""[Count how many total alleles there are]

	Args:
		l ([list]): [List of IUPAC or VCF-style (e.g. 0/1) genotypes]
		vcf (bool, optional): [Are genotypes VCF-style?]. Defaults to False.

	Returns:
		[int]: [Total number of alleles in l]
	"""
	all=list()
	for i in l:
		if vcf:
			all.extend(i.split("/"))
			#print(i.split("/"))
		else:
			all.extend(get_iupac_caseless(i))
	all = remove_items(all, ["-9", "-", "N", -9])
	return len(set(all))

def get_major_allele(l, num=None, vcf=False):
	"""[Get most common alleles in list]

	Args:
		l ([list]): [List of genotypes for one sample]
		num (int, optional): [Number of elements to return]. Defaults to None.
		vcf (bool, optional): [Alleles in VCF-style format?]. Defaults to False.

	Returns:
		[list]: [[num] most common alleles in descending order]
	"""
	all=list()
	for i in l:
		if vcf:
			all.extend(i.split("/"))
		else:
			all.extend(get_iupac_caseless(i))

	c = Counter(all) # requires collections import
	rets = c.most_common(num)

	# Returns two most common non-ambiguous bases
	# Makes sure the least common base isn't N or -9
	if vcf:
		return [x[0] for x in rets if x[0] != "-9"]
	else:
		return [x[0] for x in rets if x[0] in ["A", "T", "G", "C"]]
	
def get_iupac_caseless(char):
	"""[Split IUPAC code to two primary characters, assuming diploidy; gives all non-valid ambiguities as N]

	Args:
		char ([str]): [Base to expand into diploid list]

	Returns:
		[list]: [List of the two expanded alleles]
	"""
	lower = False
	if char.islower():
		lower = True
		char = char.upper()
	iupac = {
		"A"	: ["A","A"],
		"G"	: ["G","G"],
		"C"	: ["C","C"],
		"T"	: ["T","T"],
		"N"	: ["N","N"],
		"-"	: ["N","N"],
		"R"	: ["A","G"],
		"Y"	: ["C","T"],
		"S"	: ["G","C"],
		"W"	: ["A","T"],
		"K"	: ["G","T"],
		"M"	: ["A","C"],
		"B"	: ["N","N"],
		"D"	: ["N","N"],
		"H"	: ["N","N"],
		"V"	: ["N","N"]
	}
	ret = iupac[char]
	if lower:
		ret = [c.lower() for c in ret]
	return ret

#Function to expand ambiguous sequences
#Generator function
def expandAmbiquousDNA(sequence):
   for i in product(*[get_iupac_caseless(j) for j in sequence]):
      yield("".join(i))

#Function to return reverse complement of a nucleotide, while preserving case
def get_revComp_caseless(char):
	lower = False
	if char.islower():
		lower = True
		char = char.upper()
	d = {
		"A"	: "T",
		"G"	: "C",
		"C"	: "G",
		"T"	: "A",
		"N"	: "N",
		"-"	: "-",
		"R"	: "Y",
		"Y"	: "R",
		"S"	: "S",
		"W"	: "W",
		"K"	: "M",
		"M"	: "K",
		"B"	: "V",
		"D"	: "H",
		"H"	: "D",
		"V"	: "B"
	}
	ret = d[char]
	if lower:
		ret = ret.lower()
	return ret


#Function to reverse complement a sequence, with case preserved
def reverseComplement(seq):
	comp = []
	for i in (get_revComp_caseless(j) for j in seq):
		comp.append(i)
	return("".join(comp[::-1]))


#Function to simplify a sequence
def simplifySeq(seq):
	temp = re.sub('[ACGT]', '', (seq).upper())
	return temp.translate(str.maketrans("RYSWKMBDHV", "**********"))

#returns dict of character counts
def seqCounter(seq):
	d = {}
	d = {
		'A':0,
		'N':0,
		'-':0,
		'C':0,
		'G':0,
		'T':0,
		"R"	: 0,
		"Y"	: 0,
		"S"	: 0,
		"W"	: 0,
		"K"	: 0,
		"M"	: 0,
		"B"	: 0,
		"D"	: 0,
		"H"	: 0,
		"V"	: 0
	}
	for c in seq:
		if c in d:
			d[c] += 1
	d['VAR'] = d['R'] + d['Y'] + d['S'] + d['W'] \
	+ d['K'] + d['M'] + d['B'] + d['D'] + d['H'] + d['V']
	return d

#Function to get vars, gaps, and N counts for flanking regions of a substring
def getFlankCounts(ref, x, y, dist):
	x2 = x-dist
	if x2 < 0:
		x2 = 0
	y2 = y+dist
	if y2 > len(ref):
		y2 = len(ref)
	flanks=ref[x2:x]+ref[y:y2] #flanks = right + left flank 
	counts=seqCounterSimple(simplifySeq(flanks))
	return(counts)


#Returns dict of character counts from a simplified consensus sequence
def seqCounterSimple(seq):
	d = {}
	d = {
		'N':0,
		'-':0,
		'*':0
	}
	for c in seq:
		if c in d:
			d[c] += 1
	return d

#Function to get GC content of a provided sequence
def gc_counts(string):
	new = re.sub('[GCgc]','#',string)
	return sum(1 for c in new if c == '#')

#Function to get counts of masked bases
def mask_counts(string):
	return sum(1 for c in string if c.islower())


#Function to get GC content as proportion
def gc_content(string):
	new = re.sub('[GCgc]','#',string)
	count = sum(1 for c in new if c == '#')
	return(count/(len(string)))

#Function to count number of lower case in a string
def mask_content(string):
	count = sum(1 for c in string if c.islower())
	return(count/(len(string)))

#generator to create sliding windows by slicing out substrings
def seqSlidingWindowString(seq, shift, width):
	seqlen = len(seq)
	for i in range(0,seqlen,shift):
		if i+width > seqlen:
			j = seqlen
		else:
			j = i + width
		yield seq[i:j]
		if j==seqlen: break

#generator to create sliding windows by slicing out substrings, returns substring indices
def seqSlidingWindow(seq, shift, width):
	seqlen = len(seq)
	for i in range(0,seqlen,shift):
		if i+width > seqlen:
			j = seqlen
		else:
			j = i + width
		yield [seq[i:j], i, j]
		if j==seqlen: break

#Function implementing fast way to replace single char in string
#This way is a lot faster than doing it by making a list and subst in list
def stringSubstitute(s, pos, c):
	return(s[:pos] + c + s[pos+1:])

#Function to return sorted unique string from list of chars
def listToSortUniqueString(l):
	sl = sorted(set(l))
	return(str(''.join(sl)))

#Function to count number of lower case in a string
def n_lower_chars(string):
    return sum(1 for c in string if c.islower())


#Function to simplify a sequence to SNP, gaps, and Ns and get counts of sliding windows
def countSlidingWindow(seq, shift, width):
	seq_temp = re.sub('[ACGT]', '', seq.upper())
	seq_norm = seq_temp.translate(str.maketrans("RYSWKMBDHV", "**********"))
	for i in windowSub(seq_norm, shift, width):
		#print(i)
		window_seq = "".join(i)
		seqCounterSimple(window_seq)

#Object for creating an iterable slidinw window sampling
class slidingWindowGenerator():
	#Need to come back and comment better...
	def __init__(self, seq, shift, width):
		self.__seq = seq
		self.__seqlen = len(self.__seq)
		self.__shift = shift
		self.__width = width
		self.__i = 0

	def __call__(self):
		self.__seqlen
		while self.__i < self.__seqlen:
			#print("i is ", self.__i, " : Base is ", self.__seq[self.__i]) #debug print
			if self.__i+self.__width > self.__seqlen:
				j = self.__seqlen
			else:
				j = self.__i + self.__width
			yield [self.__seq[self.__i:j], self.__i, j]
			if j==self.__seqlen: break