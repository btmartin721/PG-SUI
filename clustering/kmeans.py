
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScalar

class RunKmeans:

	def __init__(self, data):
		self.data = data

	def 