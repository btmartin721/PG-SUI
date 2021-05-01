# Standard library imports
import sys
import os
import math
from collections import OrderedDict

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import accuracy_score

# Custom module imports
from utils import misc

def impute_knn(data, knn_settings):
	"""[Impute missing data using the K-nearest neighbors algorithm]

	Raises:
		ValueError: [If a KNNImputer is not supported]
		AssertionError: [If there's any missing data left]

	Returns:
		[pandas.DataFrame]: [imputed dataframe]
	"""
	df = pd.DataFrame.from_records(data)

	df.replace(-9, pd.NA, inplace=True)

	for col in df:
		df[col] = df[col].astype("Int8")

	imputer = KNNImputer(n_neighbors=knn_settings["n_neighbors"], weights=knn_settings["weights"], metric=knn_settings["metric"])

	df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

	# If there are any remaining missing data raise error.
	if any(df.isna().any().to_list()) == True:
		raise AssertionError("\nThere was a problem with the K-NN imputation. Please inspect your data and try again.")

	for col in df:
		df[col] = df[col].astype(int)

	return df

def impute_knn_optk(snpslist, popslist, knn_settings, maxk, np):
	"""[Run K-nearest neighbors with n_neighbors ranging from 1 to maxk]

	Args:
		snpslist ([list(list)]): [Snps object from GenotypeData]
		popslist ([list]): [Population IDs from GenotypeData object]
		knn_settings ([dict]): [Settings for KNNImputer]
		maxk ([int]): [Maximum n_neighbors value to test K-NN with]
		np ([int]): [Number of processors to use for optimization]

	Returns:
		[int]: [optimal n_neighbors value with lowest RMSE score]
	"""
	df_X = pd.DataFrame.from_records(snpslist)
	df_X.replace(-9, pd.NA, inplace=True)

	for col in df_X:
		df_X[col] = df_X[col].astype("Int8")


	le = LabelEncoder()
	pops_encoded = le.fit_transform(popslist)
	pops_y = pd.Series(pops_encoded)

	#RANDOM_STATE = 456
	errors = list()
	accuracy = list()
	for k in range(1, int(maxk)):
		imputer = KNNImputer(n_neighbors=k)
		imputed = imputer.fit_transform(df_X)
		df_imputed = pd.DataFrame(imputed, columns=df_X.columns)

		X_train, X_test, y_train, y_test = train_test_split(df_imputed, pops_y, test_size=0.2)
		
		# NOTE: The below code is for evaluating the random forest model to 
		# obtain the best parameters.

		# NOTE: Setting the `warm_start` construction parameter to `True` 
		# disables support for parallelized ensembles but is necessary for 
		# tracking the OOB error trajectory during training.
		# ensemble_clfs = [
		# 	("RandomForestClassifier, max_features='sqrt'",
		# 		RandomForestClassifier(warm_start=True, oob_score=True,
		# 							max_features='sqrt',
		# 							random_state=RANDOM_STATE)),
		# 	("RandomForestClassifier, max_features='log2'",
		# 		RandomForestClassifier(warm_start=True, oob_score=True,
		# 							max_features="log2",
		# 							random_state=RANDOM_STATE)),
		# 	("RandomForestClassifier, max_features=None",
		# 		RandomForestClassifier(warm_start=True, oob_score=True,
		# 							max_features=None,
		# 							random_state=RANDOM_STATE)),
		# ]

		# Map a classifier name to a list of (<n_estimators>, <error rate>) 
		# pairs.
		#error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)


		# Range of `n_estimators` values to explore.
		#min_estimators = 50
		#max_estimators = 1000

		# for label, clf in ensemble_clfs:
		# 	for i in range(min_estimators, max_estimators + 1, 50):
		# 		clf.set_params(n_estimators=i)
		# 		clf.fit(X_train, y_train)

		# 		# Record the OOB error for each `n_estimators=i` setting.
		# 		oob_error = 1 - clf.oob_score_
		# 		error_rate[label].append((i, oob_error))

		# # Generate the "OOB error rate" vs. "n_estimators" plot.
		# for label, clf_err in error_rate.items():
		# 	xs, ys = zip(*clf_err)
		# 	plt.plot(xs, ys, label=label)

		# plt.xlim(min_estimators, max_estimators)
		# plt.xlabel("n_estimators")
		# plt.ylabel("OOB error rate")
		# plt.legend(loc="upper right")
		# plt.show()
		# sys.exit()

		# Classifier to evaluate the best n_neighbors (K)
		model = RandomForestClassifier(n_estimators=500, 
										max_features="log2", 
										n_jobs = np)

		model.fit(X_train, y_train)
		preds = model.predict(X_test)

		# import sklearn.metrics.mean_squared_error()
		error = rmse(y_test, preds) 
		acc = accuracy_score(y_test, preds)

		errors.append({'K': k, 'RMSE': error, "ACC": acc})

	# Get the best RMSE score
	optimalk, test_acc = get_lowest_rmse(errors)

	return optimalk, test_acc

def get_lowest_rmse(rmseerrors):
	"""[Gets the "K" value with the lowest "RMSE" value]

	Args:
		rmseerrors ([list(dict)]): [Object returned from rmse]

	Returns:
		[int]: [K value with the lowest RMSE score]
	"""
	minrmse = None
	optk = None
	for item in rmseerrors:
		rmseval = item["RMSE"]
		acceval = item["ACC"]
		k = item["K"]
		if minrmse:
			if rmseval < minrmse:
				optk = k
				minrmse = rmseval
				bestacc = acceval
			else:
				continue
		else:
			optk = k
			minrmse = item["RMSE"]
			bestacc = item["ACC"]

	return optk, bestacc
    
def most_common(mylist):
	"""[Get the most frequent value in a list]

	Args:
		mylist ([list]): [list of values]

	Returns:
		[int]: [Most frequent integer in the list]
	"""
	counter = 0
	num = mylist[0]

	for i, k in enumerate(mylist):
		curr_frequency = mylist.count(k)
		if curr_frequency > counter:
			counter = curr_frequency
			num = k
	final_count = mylist.count(num)
	reps = len(mylist)
	return num, i, final_count, reps

@misc.timer
def rf_imputer(snpslist, settings):
	"""[Do random forest imputation using Iterative Imputer. Iterative imputer iterates over all the other features (columns) and uses each one as a target variable, thereby informing missingness in the input column]

	Args:
		snpslist ([list(list)]): [012-encoded genotypes from GenotypeData]
		settings ([dict]): [Keys as setting arguments, values as setting values]

	Returns:
		[numpy.ndarray]: [2-D numpy array with imputed genotypes]
	"""
	print("\nDoing random forest imputation...\n")
	print(
			"""
			Random Forest Classifier Settings:
				rf_n_estimators: """+str(settings["rf_n_estimators"])+"""
				rf_min_samples_leaf: """+str(settings["rf_min_samples_leaf"])+"""
				rf_n_jobs: """+str(settings["rf_n_jobs"])+"""
				rf_max_features: """+str(settings["rf_max_features"])+"""
				rf_criterion: """+str(settings["rf_criterion"])+"""
				rf_random_state: """+str(settings["rf_random_state"])+"""
				rf_verbose: """+str(settings["rf_verbose"])+"""
			
			Iterative Imputer Settings:
				initial strategy: """+str(settings["initial_strategy"])+"""
				max_iter: """+str(settings["max_iter"])+""" 
				random_state: """+str(settings["random_state"])+""" 
				tol: """+str(settings["tol"])+"""
				n_nearest_features: """+str(settings["n_nearest_features"])+"""
				imputation_order: """+str(settings["imputation_order"])+"""
				verbose: """+str(settings["verbose"])+"""
			"""
	)

	df = format_features(snpslist)

	# Create iterative imputer
	imputed = IterativeImputer(
		estimator=ExtraTreesClassifier(
							n_estimators=settings["rf_n_estimators"],
							min_samples_leaf=settings["rf_min_samples_leaf"], 
							n_jobs=settings["rf_n_jobs"],
							max_features=settings["rf_max_features"],
							criterion=settings["rf_criterion"],
							random_state=settings["rf_random_state"],
							verbose=settings["rf_verbose"]),
		initial_strategy=settings["initial_strategy"],
		max_iter=settings["max_iter"], 
		random_state = settings["random_state"], 
		tol=settings["tol"], 
		n_nearest_features=settings["n_nearest_features"], 
		imputation_order=settings["imputation_order"],
		verbose=settings["verbose"]
		)

	arr = imputed.fit_transform(df)

	new_arr = arr.astype(dtype=np.int)

	print("\nDone!")

	return new_arr

@misc.timer
def gb_imputer(snpslist, settings):
	"""[Do gradient boosting imputation using Iterative Imputer.
	Iterative imputer iterates over all the other features (columns)
	and uses each one as a target variable, thereby informing missingness
	in the input column]

	Args:
		snpslist ([list(list)]): [012-encoded genotypes from GenotypeData]
		settings ([dict]): [Keys as setting arguments, values as setting values]

	Returns:
		[numpy.ndarray]: [2-D numpy array with imputed genotypes]
	"""
	print("\nDoing gradient boosting iterative imputation...\n")
	print(
			"""
			Gradient Boosting Classifier Settings:
				gb_n_estimators: """+str(settings["gb_n_estimators"])+"""
				gb_min_samples_leaf: """+str(settings["gb_min_samples_leaf"])+"""
				gb_learning_rate: """+str(settings["gb_learning_rate"])+"""
				gb_max_features: """+str(settings["gb_max_features"])+"""
				gb_criterion: """+str(settings["gb_criterion"])+"""
				gb_subsample: """+str(settings["gb_subsample"])+"""
				gb_loss: """+str(settings["gb_loss"])+"""
				gb_min_samples_split: """+str(settings["gb_min_samples_split"])+"""
				gb_max_depth: """+str(settings["gb_max_depth"])+"""
				gb_validation_fraction: """+str(settings["gb_validation_fraction"])+"""
				gb_n_iter_no_change: """+str(settings["gb_n_iter_no_change"])+"""
				gb_tol: """+str(settings["gb_tol"])+"""
				gb_random_state: """+str(settings["gb_random_state"])+"""
				gb_verbose: """+str(settings["gb_verbose"])+"""
			
			Iterative Imputer Settings:
				initial strategy: """+str(settings["initial_strategy"])+"""
				max_iter: """+str(settings["max_iter"])+""" 
				random_state: """+str(settings["random_state"])+""" 
				tol: """+str(settings["tol"])+"""
				n_nearest_features: """+str(settings["n_nearest_features"])+"""
				imputation_order: """+str(settings["imputation_order"])+"""
				verbose: """+str(settings["verbose"])+"""
			"""
	)

	df = format_features(snpslist)

	# Create iterative imputer
	imputed = IterativeImputer(
		estimator=GradientBoostingClassifier(
						n_estimators=settings["gb_n_estimators"], 
						min_samples_leaf=settings["gb_min_samples_leaf"],
						max_features=settings["gb_max_features"],
						learning_rate=settings["gb_learning_rate"],
						criterion=settings["gb_criterion"],
						subsample=settings["gb_subsample"],
						loss=settings["gb_loss"],
						min_samples_split=settings["gb_min_samples_split"],
						max_depth=settings["gb_max_depth"],
						random_state=settings["gb_random_state"],
						verbose=settings["gb_verbose"],
						validation_fraction=settings["gb_validation_fraction"],
						n_iter_no_change=settings["gb_n_iter_no_change"],
						tol=settings["gb_tol"]
					),
		initial_strategy=settings["initial_strategy"],
		max_iter=settings["max_iter"], 
		random_state = settings["random_state"], 
		tol=settings["tol"], 
		n_nearest_features=settings["n_nearest_features"], 
		imputation_order=settings["imputation_order"],
		verbose=settings["verbose"]
	)

	arr = imputed.fit_transform(df)

	new_arr = arr.astype(dtype=np.int)

	print("\nDone!")

	return new_arr

@misc.timer
def bayesianridge_imputer(snpslist, settings):
	"""[Do bayesian ridge imputation using Iterative Imputer.
	Iterative imputer iterates over all the other features (columns)
	and uses each one as a target variable, thereby informing missingness
	in the input column]

	Args:
		snpslist ([list(list)]): [012-encoded genotypes from GenotypeData]
		settings ([dict]): [Keys as setting arguments, values as setting values]

	Returns:
		[numpy.ndarray]: [2-D numpy array with imputed genotypes]
	"""
	print("\nDoing bayesian ridge iterative imputation...\n")
	print(
			"""
			Bayesian Ridge Regression Settings:
				br_n_iter: """+str(settings["br_n_iter"])+"""
				br_tol: """+str(settings["br_tol"])+"""
				br_alpha_1: """+str(settings["br_alpha_1"])+"""
				br_alpha_2: """+str(settings["br_alpha_2"])+"""
				br_lambda_1: """+str(settings["br_lambda_1"])+"""
				br_lambda_2: """+str(settings["br_lambda_2"])+"""
				br_alpha_init: """+str(settings["br_alpha_init"])+"""
				br_lambda_init: """+str(settings["br_lambda_init"])+"""
				br_verbose: """+str(settings["br_verbose"])+"""

			Iterative Imputer Settings:
				initial strategy: """+str(settings["initial_strategy"])+"""
				max_iter: """+str(settings["max_iter"])+""" 
				random_state: """+str(settings["random_state"])+""" 
				tol: """+str(settings["tol"])+"""
				n_nearest_features: """+str(settings["n_nearest_features"])+"""
				imputation_order: """+str(settings["imputation_order"])+"""
				verbose: """+str(settings["verbose"])+"""
			"""
	)

	df = format_features(snpslist)

	# Create iterative imputer
	imputed = IterativeImputer(
		estimator=BayesianRidge(
							n_iter=settings["br_n_iter"],
							tol=settings["br_tol"],
							alpha_1=settings["br_alpha_1"],
							alpha_2=settings["br_alpha_2"],
							lambda_1=settings["br_lambda_1"],
							lambda_2=settings["br_lambda_2"],
							alpha_init=settings["br_alpha_init"],
							lambda_init=settings["br_lambda_init"],
							verbose=settings["br_verbose"]
					),
		initial_strategy=settings["initial_strategy"],
		max_iter=settings["max_iter"], 
		random_state = settings["random_state"], 
		tol=settings["tol"], 
		n_nearest_features=settings["n_nearest_features"], 
		imputation_order=settings["imputation_order"],
		verbose=settings["verbose"],
		sample_posterior=settings["br_sample_posterior"]
	)

	arr = imputed.fit_transform(df)

	new_arr = arr.astype(dtype=np.int)

	print("\nDone!")

	return new_arr

@misc.timer
def knn_iterative_imputer(snpslist, settings):
	"""[Do K-nearest neighbors imputation using Iterative Imputer.
	Iterative imputer iterates over all the other features (columns)
	and uses each one as a target variable, thereby informing missingness
	in the input column]

	Args:
		snpslist ([list(list)]): [012-encoded genotypes from GenotypeData]
		settings ([dict]): [Keys as setting arguments, values as setting values]

	Returns:
		[numpy.ndarray]: [2-D numpy array with imputed genotypes]
	"""
	print("\nDoing K-nearest neighbor iterative imputation...\n")
	print(
			"""
			K Neighbors Classifier Settings:
				knn_it_n_neighbors: """+str(settings["knn_it_n_neighbors"])+"""
				knn_it_weights: """+str(settings["knn_it_weights"])+"""
				knn_it_algorithm: """+str(settings["knn_it_algorithm"])+"""
				knn_it_leaf_size: """+str(settings["knn_it_leaf_size"])+"""
				knn_it_power: """+str(settings["knn_it_power"])+"""
				knn_it_metric: """+str(settings["knn_it_metric"])+"""
				knn_it_metric_params: """+str(settings["knn_it_metric_params"])+"""
				n_jobs: """+str(settings["knn_it_n_jobs"])+"""

			Iterative Imputer Settings:
				initial strategy: """+str(settings["initial_strategy"])+"""
				max_iter: """+str(settings["max_iter"])+""" 
				random_state: """+str(settings["random_state"])+""" 
				tol: """+str(settings["tol"])+"""
				n_nearest_features: """+str(settings["n_nearest_features"])+"""
				imputation_order: """+str(settings["imputation_order"])+"""
				verbose: """+str(settings["verbose"])+"""
				"""
	)

	df = format_features(snpslist)

	# Create iterative imputer
	imputed = IterativeImputer(
		estimator=KNeighborsClassifier(
							n_neighbors=settings["knn_it_n_neighbors"],
							weights=settings["knn_it_weights"],
							algorithm=settings["knn_it_algorithm"],
							leaf_size=settings["knn_it_leaf_size"],
							p=settings["knn_it_power"],
							metric=settings["knn_it_metric"],
							metric_params=settings["knn_it_metric_params"],
							n_jobs=settings["knn_it_n_jobs"],
					),
		initial_strategy=settings["initial_strategy"],
		max_iter=settings["max_iter"], 
		random_state = settings["random_state"], 
		tol=settings["tol"], 
		n_nearest_features=settings["n_nearest_features"], 
		imputation_order=settings["imputation_order"],
		verbose=settings["verbose"]
	)

	arr = imputed.fit_transform(df)

	new_arr = arr.astype(dtype=np.int)

	print("\nDone!")

	return new_arr

def impute_freq(data, pops=None, diploid=True, default=0, missing=-9):
	"""[Impute missing genotypes using allele frequencies, with missing alleles coded as negative; usually -9]
	
	Args:
		data ([list(list)]): [List containing list of genotypes in integer format]

		pop ([list], optional): [List of population assignments. Default is None
			When pop=None, allele frequencies are computed globally]

		diploid ([Boolean] optional): [When TRUE, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt
			When diploid=FALSE, 0,1, and 2 are sampled according to their observed frequency
			When dipoid=TRUE, 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt)
				In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes]

		default ([int, optional]): [Value to set if no alleles sampled at locus]. Default = 0
		missing ([int, optional]): [missing data value]. Default = -9

	Returns:
		[list(list)]: [Imputed genotypes of same dimensions as data]
	"""
	if pops:
		print("\nImputing by population allele frequencies...")
	else:
		print("\nImputing by global allele frequency...")

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

	print("Done!")
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

def format_features(featurelist, missing_val=-9):
	"""[Format a 2D list for input into iterative imputer]

	Args:
		featurelist ([list(list)]): [2D list of features with shape(n_samples, n_features)]
		missing_val (int, optional): [Missing value to replace with numpy.nan]. Defaults to -9.

	Returns:
		[pandas.DataFrame]: [Formatted pandas.DataFrame for input into IterativeImputer]
	"""
	# Make pandas.DataFrame from 2D list
	df = pd.DataFrame.from_records(featurelist)

	# Replace missing data with NaN
	df.replace(missing_val, np.nan, inplace=True)

	# Cast features as 8-bit integers
	for col in df:
		df[col] = df[col].astype("Int8")

	return df

