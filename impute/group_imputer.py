import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class GroupImputer(BaseEstimator, TransformerMixin):
	"""[Class used for imputing missing values in a pd.DataFrame using either mean or median of a group]
		
	Args:
	group_cols ([list]): [List of columns used for calculating the aggregated value]

	target ([str]): [The name of the column to impute]

	metric ([str]): [The metric to be used for remplacement, can be one of ['mean', 'median', 'most_frequent']]

	Returns:
	X ([array-like]): [The array with imputed values in the target column]
	"""
	def __init__(self, group_cols, target, metric="mean"):
			
		assert metric in [
			"mean", "median", "most_frequent"
		], "Unrecognized value for metric, should be mean/median/most_frequent"

		assert (
			type(group_cols) == list, "group_cols should be a list of columns"
		)

		assert type(target) == str, "Target should be a string"
			
		self.group_cols = group_cols
		self.target = target
		self.metric = metric

		if self.metric == "most_frequent":
			self.metric = pd.Series.mode
		
	def fit(self, X, y=None):
			
		assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
			
		impute_map = X.groupby(self.group_cols)[self.target].agg(
			self.metric).reset_index(drop=False)
			
		self.impute_map_ = impute_map
			
		return self 
		
	def transform(self, X, y=None):
			
		# make sure that the imputer was fitted
		check_is_fitted(self, 'impute_map_')
			
		X = X.copy()
			
		for index, row in self.impute_map_.iterrows():
			ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)

			X.loc[ind, self.target] = X.loc[ind, self.target].fillna(
				row[self.target]
			)
			
		return X.values