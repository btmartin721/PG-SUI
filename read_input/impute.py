
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