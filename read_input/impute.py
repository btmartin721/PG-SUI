import sys
import os
from utils import misc


class ImputeMissing:
	
	def __init__(self, data, method=None):
		"""[Methods for imputing missing data in a GenotypeData object]

		Args:
			data ([GenotypeData object]): [Object returned from GenotypeData]
			method ([list], optional): [Imputation methods to run]. Defaults to None.
		"""
		self.data = data
		self.method = method

	def impute_knn(self):
		pass

def impute_knn(data):
	pass

def impute_freq(data, pops=None, diploid=True, default=0, missing=-9):
	"""[Impute missing genotypes using allele frequencies, with missing alleles coded as negative; usually -9]
	
	Args:
	data ([List of lists]): List containing list of genotypes in integer format
	pop ([list] optional): List of population assignments. Default is None
		When pop=None, allele frequencies are computed globally
	diploid ([Boolean] optional): When TRUE, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt
		When diploid=FALSE, 0,1, and 2 are sampled according to their observed frequency
		When dipoid=TRUE, 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt)
			In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes
	Returns lists of lists of same dimensions as data
	"""
	bak=data
	data=[x[:] for x in bak]
	if pops is not None:
		pop_indices = misc.get_indices(pops)
	loc_index=0
	for locus in data:
		if pops is None:
			allele_probs = get_allele_probs(locus, diploid)
			#print(allele_probs)
			if misc.all_zero(list(allele_probs.values())) or not allele_probs:
				print("\nWarning: No alleles sampled at locus",str(loc_index),"setting all values to:",str(default))
				gen_index=0
				for geno in locus:
					data[loc_index][gen_index] = default
					gen_index+=1
			else:
				gen_index=0
				for geno in locus:
					if geno == missing:
						data[loc_index][gen_index] = sample_allele(allele_probs, diploid=True)
					gen_index+=1
					
		else:
			for pop in pop_indices.keys():
				allele_probs = get_allele_probs(locus, diploid, missing=missing, indices=pop_indices[pop])
				#print(pop,"--",allele_probs)
				if misc.all_zero(list(allele_probs.values())) or not allele_probs:
					print("\nWarning: No alleles sampled at locus",str(loc_index),"setting all values to:",str(default))
					gen_index=0
					for geno in locus:
						data[loc_index][gen_index] = default
						gen_index+=1
				else:
					gen_index=0
					for geno in locus:
						if geno == missing:
							data[loc_index][gen_index] = sample_allele(allele_probs, diploid=True)
						gen_index+=1
				
		loc_index+=1
	return(data)

def sample_allele(allele_probs, diploid=True):
	if diploid:
		alleles=misc.weighted_draw(allele_probs, 2)
		if alleles[0] == alleles[1]:
			return(alleles[0])
		else:
			return(1)
	else:
		return(misc.weighted_draw(allele_probs, 1)[0])

def get_allele_probs(genotypes, diploid=True, missing=-9, indices=None):
	data=genotypes
	length=len(genotypes)
	
	if indices is not None:
		data = [genotypes[index] for index in indices]
		length = len(data)
	
	if len(set(data))==1:
		if data[0] == missing:
			ret=dict()
			return(ret)
		else:
			ret=dict()
			ret[data[0]] = 1.0
			return(ret)
	
	if diploid:
		length = length*2
		ret = {0:0.0, 2:0.0}
		for g in data:
			if g == 0:
				ret[0] += 2
			elif g == 2:
				ret[2] += 2
			elif g == 1:
				ret[0] += 1
				ret[2] += 1
			elif g == missing:
				length -= 2
			else:
				print("\nWarning: Ignoring unrecognized allele",str(g),"in get_allele_probs\n")
		for allele in ret.keys():
			ret[allele] = ret[allele]/float(length)
		return(ret)
	else:
		ret=dict()
		for key in set(data):
			if key != missing:
				ret[key] = 0.0
		for g in data:
			if g == missing:
				length -= 1
			else:
				ret[g] += 1
		for allele in ret.keys():
			ret[allele] = ret[allele]/float(length)
		return(ret)

def impute_common(data, pops=None):
	pass