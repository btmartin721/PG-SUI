import xgboost as xgb

class RunXGBoost:

	def __init__(self, data):
		"""[Class constructor]

		Args:
			data ([dict(list)]): [Object containing data and metadata]
		"""
		self.data = data

	def xgboost_version(self):
		print(xgb.__version__)

	