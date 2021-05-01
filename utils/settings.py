
"""[Supported and default settings for the various imputation, dimension reduction, and machine learning algorithms]
"""

def supported_imputation_settings():
	"""[Settings supported by all the imputation methods]

	Returns:
		[list and list]: [Lists of all the possible settings arguments]
	"""
	supported_settings = [
		"n_neighbors", 
		"weights", 
		"metric", 
		"rf_n_estimators",
		"rf_min_samples_leaf",
		"rf_max_features",
		"rf_n_jobs",
		"rf_criterion",
		"rf_random_state", 
		"max_iter", 
		"tol", 
		"n_nearest_features", 
		"initial_strategy", 
		"imputation_order", 
		"skip_complete", 
		"random_state",
		"verbose",
		"gb_n_estimators",
		"gb_min_samples_leaf",
		"gb_max_features",
		"gb_criterion",
		"gb_learning_rate",
		"gb_subsample",
		"gb_loss",
		"gb_min_samples_split",
		"gb_max_depth",
		"gb_random_state",
		"gb_verbose",
		"gb_validation_fraction",
		"gb_n_iter_no_change",
		"gb_tol",
		"br_n_iter",
		"br_tol",
		"br_alpha_1",
		"br_alpha_2",
		"br_lambda_1",
		"br_lambda_2",
		"br_verbose",
		"br_alpha_init",
		"br_lambda_init",
		"br_sample_posterior",
		"knn_it_n_neighbors",
		"knn_it_weights",
		"knn_it_algorithm",
		"knn_it_leaf_size",
		"knn_it_power",
		"knn_it_metric",
		"knn_it_metric_params",
		"knn_it_n_jobs"
	]

	# For K-optimization with non-iterative K-NN
	supported_settings_opt = [
		"weights", 
		"metric", 
		"reps"
	]

	return supported_settings, supported_settings_opt

def knn_imp_defaults_noniterative(maxk):
	"""[Default settings for non-iterative K-NN]

	Args:
		maxk ([int]): [Maximum K-value for K-NN optimization. If not set, defaults to None].

	Returns:
		[dict]: [Dictionary with default settings for non-iterative sklearn.impute.KNNImputer. The keys are the KNNImputer setting arguments with the corresponding default values]
	"""
	if maxk:
		knn_settings = {
			"weights": "uniform", 
			"metric": "nan_euclidean", 
			"reps": 1
		}
	else:
		knn_settings = {
			"n_neighbors": 5,
			"weights": "uniform", 
			"metric": "nan_euclidean"
		}

	return knn_settings

def knn_imp_defaults_iterative():
	"""[Default settings for iterative K-NN]

	Returns:
		[dict]: [Dictionary with sklearn.impute.KNNImputer argument names as keys with their corresponding values]
	"""
	knn_iterative_settings = {
		"knn_it_n_neighbors": 5,
		"knn_it_weights": "uniform",
		"knn_it_algorithm": "auto",
		"knn_it_leaf_size": 30,
		"knn_it_power": 2,
		"knn_it_metric": "minkowski",
		"knn_it_metric_params": None,
		"knn_it_n_jobs": 1,
		"max_iter": 10,
		"tol": 1e-3,
		"n_nearest_features": None,
		"initial_strategy": "most_frequent",
		"imputation_order": "ascending",
		"skip_complete": False,
		"random_state": None,
		"verbose": 0
	}

	return knn_iterative_settings

def random_forest_imp_defaults():
	"""[Default settings for random forest imputation]

	Returns:
		[dict]: [Dictionary with sklearn.ensemble.RandomForestClassifier argument names as keys with their corresponding values]
	"""
	rf_settings = {
			"rf_n_estimators": 100,
			"rf_min_samples_leaf": 1,
			"rf_max_features": "auto",
			"rf_n_jobs": 1,
			"rf_criterion": "gini",
			"rf_random_state": None,
			"max_iter": 10,
			"tol": 1e-3,
			"n_nearest_features": None,
			"initial_strategy": "most_frequent",
			"imputation_order": "ascending",
			"skip_complete": False,
			"random_state": None,
			"verbose": 0
	}

	return rf_settings

def gradient_boosting_imp_defaults():
	"""[Default settings for gradient boosting imputation]

	Returns:
		[dict]: [Dictionary with sklearn.ensemble.GradientBoostingClassifier argument names as keys with their corresponding values]
	"""
	gb_settings = {
			"gb_n_estimators": 100,
			"gb_min_samples_leaf": 1,
			"gb_max_features": "auto",
			"gb_criterion": "friedman_mse",
			"gb_learning_rate": 0.1,
			"gb_subsample": 1.0,
			"gb_loss": "deviance",
			"gb_min_samples_split": 2,
			"gb_max_depth": 3,
			"gb_random_state": None,
			"gb_verbose": 0,
			"gb_validation_fraction": 0.1,
			"gb_n_iter_no_change": None,
			"gb_tol": 1e-4,
			"max_iter": 10,
			"tol": 1e-3,
			"n_nearest_features": None,
			"initial_strategy": "most_frequent",
			"imputation_order": "ascending",
			"skip_complete": False,
			"verbose": 0,
			"random_state": None
	}

	return gb_settings

def bayesian_ridge_imp_defaults():
	"""[Default settings for gradient boosting imputation]

	Returns:
		[dict]: [Dictionary with sklearn.linear_model.BayesianRidge argument names as keys with their corresponding values]
	"""
	br_settings = {
			"br_n_iter": 300,
			"br_tol": 1e-3,
			"br_alpha_1": 1e-6,
			"br_alpha_2": 1e-6,
			"br_lambda_1": 1e-6,
			"br_lambda_2": 1e-6,
			"br_verbose": False,
			"br_alpha_init": None,
			"br_lambda_init": None,
			"br_sample_posterior": False,
			"max_iter": 10,
			"tol": 1e-3,
			"n_nearest_features": None,
			"initial_strategy": "most_frequent",
			"imputation_order": "ascending",
			"skip_complete": False,
			"random_state": None,
			"verbose": 0
	}

	return br_settings

def random_forest_embed_supported():
	"""[Supported argument settings for random forest embedding]

	Returns:
		[list]: [Supported argument names for sklearn.ensemble.RandomTreesEmbedding]
	"""
	supported_settings = [
		"rf_n_estimators"
		"rf_max_depth",
		"rf_min_samples_split",
		"rf_min_samples_leaf",
		"rf_min_weight_fraction_leaf",
		"rf_max_leaf_nodes",
		"rf_min_impurity_decrease",
		"rf_min_impurity_split",
		"rf_sparse_output",
		"rf_n_jobs",
		"rf_random_state",
		"rf_verbose",
		"rf_warm_start"
	]

	return supported_settings

def random_forest_embed_defaults():
	"""[Default settings for random forest embedding]

	Returns:
		[dict]: [Dictionary with sklearn.ensemble.RandomTreesEmbedding argument names as keys with their corresponding values]
	"""
	rf_settings_default = {
		"rf_n_estimators": 100,
		"rf_max_depth": 5,
		"rf_min_samples_split": 2,
		"rf_min_samples_leaf": 1,
		"rf_min_weight_fraction_leaf": 0.0,
		"rf_max_leaf_nodes": None,
		"rf_min_impurity_decrease": 0.0,
		"rf_min_impurity_split": None,
		"rf_sparse_output": True,
		"rf_n_jobs": None,
		"rf_random_state": None,
		"rf_verbose": 0,
		"rf_warm_start": False
	}

	return rf_settings_default

def dim_reduction_supported_arguments():
	"""[Supported settings for dimensionality reduction algorithms]

	Returns:
		[list]: [Supported argument names for the various dimensionality reduction algorithms]
	"""
	supported_settings = [
		"n_components", 
		"copy", 
		"scaler", 
		"ploidy",
		"pc_axis1",
		"pc_axis2",
		"figwidth", 
		"figheight", 
		"alpha", 
		"legend", 
		"legend_inside", 
		"legend_loc", 
		"marker", 
		"markersize", 
		"markeredgecolor", 
		"markeredgewidth", 
		"labelspacing", 
		"columnspacing", 
		"title", 
		"title_fontsize",
		"markerfirst", 
		"markerscale", 
		"labelcolor", 
		"ncol", 
		"bbox_to_anchor", 
		"borderaxespad", 
		"legend_edgecolor", 
		"facecolor", 
		"framealpha", 
		"shadow",
		"n_dims",
		"random_state",
		"n_init",
		"max_iter",
		"verbose",
		"eps",
		"n_jobs",
		"dissimilarity",
		"cmds_axis1",
		"cmds_axis2",
		"isomds_axis1",
		"isomds_axis2"
	]

	return supported_settings

def dim_reduction_supported_algorithms():
	"""[Supported algorithms for dimensionality reduction]

	Returns:
		[list]: [List of supported algorithms]
	"""
	return ["standard-pca", "cmds", "isomds"]

def pca_default_settings():
	"""[Default settings for standard PCA]

	Returns:
		[dict]: [Dictionary with sklearn.decomposition.PCA argument names as keys with their corresponding values]
	"""
	pca_settings_default = {
		"n_components": 10, 
		"copy": True, 
		"scaler": "patterson", 
		"ploidy": 2,
		"pc_axis1": 1,
		"pc_axis2": 2,
		"figwidth": 6, 
		"figheight": 6, 
		"alpha": 1.0, 
		"legend": True, 
		"legend_inside": False, 
		"legend_loc": "upper left", 
		"marker": 'o', 
		"markersize": 6, 
		"markeredgecolor": 'k', 
		"markeredgewidth": 0.5, 
		"labelspacing": 0.5, 
		"columnspacing": 2.0, 
		"title": None, 
		"title_fontsize": None,
		"markerfirst": True, 
		"markerscale": 1.0, 
		"labelcolor": "black", 
		"ncol": 1, 
		"bbox_to_anchor": (1.0, 1.0), 
		"borderaxespad": 0.5, 
		"legend_edgecolor": "black", 
		"facecolor": "white", 
		"framealpha": 0.8, 
		"shadow": False
	}

	return pca_settings_default

def mds_default_settings():
	"""[Default settings for metric and non-metric multidimensional scaling (MDS)]

	Returns:
		[dict]: [Dictionary with sklearn.decomposition.PCA argument names as keys with their corresponding values]
	"""
	mds_settings_default = {
	"n_dims": 2, 
	"random_state": None, 
	"n_init": 4,
	"max_iter": 300,
	"verbose": 0,
	"eps": 1e-3,
	"n_jobs": 1,
	"dissimilarity": "euclidean",
	"mds_axis1": 1,
	"mds_axis2": 2,
	"figwidth": 6, 
	"figheight": 6, 
	"alpha": 1.0, 
	"legend": True, 
	"legend_inside": False, 
	"legend_loc": "upper left", 
	"marker": 'o', 
	"markersize": 6, 
	"markeredgecolor": 'k', 
	"markeredgewidth": 0.5, 
	"labelspacing": 0.5, 
	"columnspacing": 2.0, 
	"title": None, 
	"title_fontsize": None,
	"markerfirst": True, 
	"markerscale": 1.0, 
	"labelcolor": "black", 
	"ncol": 1, 
	"bbox_to_anchor": (1.0, 1.0), 
	"borderaxespad": 0.5, 
	"legend_edgecolor": "black", 
	"facecolor": "white", 
	"framealpha": 0.8, 
	"shadow": False,
	"cmds_axis1": 1,
	"cmds_axis2": 2,
	"isomds_axis1": 1,
	"isomds_axis2": 2
	}
	
	return mds_settings_default



