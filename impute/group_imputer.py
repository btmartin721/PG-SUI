import sys

import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.base import BaseEstimator, TransformerMixin

class ImputeByGroup(BaseEstimator, TransformerMixin):
	def __init__(self, group_var):
		self.group_var = group_var
		
	def fit(self, X, y=None):
		return self
			
	def transform(self, X):
		# the copy leaves the original dataframe intact
		X_ = X.copy()
		for col in X_.columns:
			if X_[col].dtypes == 'float64':
				X_.loc[(X[col].isna()) & X_[self.group_var].notna(), col] = X_[self.group_var].map(X_.groupby(self.group_var)[col].mean())
				X_[col] = X_[col].fillna(X_[col].mean())
		return X_