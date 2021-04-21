
import allel

import numpy as np

class DimReduction:

	def __init__(self, data, algorithms=None, settings=None):
		self.data = data
		self.settings = settings
		self.algorithms = algorithms

		self.all_settings = {"n_components": 10, 
									"copy": True, 
									"scaler": "patterson", 
									"ploidy": 2
								}

		if self.algorithms:
			self.algorithms = [self.algorithms]

		for arg in self.algorithms:
			if arg not in ["standard-pca"]:
				raise ValueError("\nThe argument {} is not supported. Supported options include: [standard-pca]".format(arg))

		if self.settings and arg == "standard-pca":
			self.all_settings.update(self.settings)
			self.standard_pca(self.all_settings)
		else:
			self.standard_pca(self.all_settings)

		

	def standard_pca(self, all_settings):

		pca_arguments = ["n_components", "copy", "scaler", "ploidy"]

		# Subset all_settings to only keys in pca_arguments
		pca_arguments = {key: all_settings[key] for key in pca_arguments}

		gn = np.array(self.data).transpose()
		print(gn)

		#res = allel.pca(gn, n_components=pca_arguments["n_components"], copy=pca_arguments["copy"], scaler=pca_arguments["scaler"], ploidy=pca_arguments["ploidy"])
