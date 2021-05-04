
class ModelResult:
"""[Object to hold results from replicates of a single delim model]
	
"""
def __init__(self):
	self.model = None
	self.reps = list()
	self.num_reps = 0
	
def clusterAcrossK(self, method, inplace=True):
	ret=list()
	for i in range(0, len(self.reps)):
		if method == "exhaustive":
			#ret.append()
			pass
		elif method == "heuristic":
			#ret.append()
			pass
		elif method == "graph":
			#ret.append()
			pass
		else:
			sys.exit("\nError: Method",method,"invalid for ModelResult.clusterAcrossK\n")
		
	if inplace:
		for i in range(0, len(ret)):
			self.reps[i] = reps[i]
	else:
		return(ret)
	