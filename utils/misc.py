import sys
import os

from numpy.random import choice

"""
Takes a list and returns dict giving indices matching each possible list member
Example:
	Input [0, 1, 1, 0, 0]
	Output {0:[0,3,4], 1:[1,2]}
"""
def get_indices(l):
	ret=dict()
	for member in set(l):
		ret[member] = list()
	i=0
	for el in l:
		ret[el].append(i)
		i+=1
	return(ret)

"""
Returns TRUE if supplied list contains all zeros
Returns FALSE if list contains ANY non-zero values
Returns FALSE if list is empty
"""
def all_zero(l):
	values=set(l)
	if len(values) > 1:
		return(False)
	elif len(values)==1 and l[0] in [0, 0.0, "0", "0.0"]:
		return(True)
	else:
		return(False)

def weighted_draw(d, num_samples=1):
	choices = list(d.keys())
	weights = list(d.values())
	return(choice(choices, num_samples, p=weights))