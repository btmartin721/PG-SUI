
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

def random_forest_unsupervised_supported():
	"""[Supported argument settings for random forest embedding]

	Returns:
		[list]: [Supported argument names for sklearn.ensemble.RandomTreesEmbedding]
	"""
	supported_settings = [
		"rf_n_estimators",
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

def random_forest_unsupervised_defaults():
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
	"ncol": 1, 
	"bbox_to_anchor": (1.0, 1.0), 
	"borderaxespad": 0.5, 
	"legend_edgecolor": "black", 
	"facecolor": "white", 
	"framealpha": 0.8, 
	"shadow": False,
	"cmds_axis1": 1,
	"cmds_axis2": 2,
	"cmds_axis3": 3,
	"isomds_axis1": 1,
	"isomds_axis2": 2,
	"isomds_axis3": 3,
	"mds_axis1": 1,
	"mds_axis2": 2,
	"mds_axis3": 3
	}
	
	return mds_settings_default

def dimreduction_plot_settings():
	"""[Default settings for dimensionality reduction plots]

	Returns:
		[dict]: [Argment names as keys with their corresponding values. See matplotlib documentation]

	Options:
		axis1 (int, optional): [First axis to plot; starts at 1]. Defaults to 1.

		axis2 (int, optional): [Second axis to plot; starts at 1]. Defaults to 2.

		axis3 (int, optional): [Third axis to plot; starts at 1]. Defaults to 3.

		figwidth (int, optional): [Set width of the plot]. Defaults to 6.

		figheight (int, optional): [Set height of the plot]. Defaults to 6.

		alpha (float, optional): [Set transparency of sample points. Should be between 0 and 1. 0 = fully transparent, 1 = no transparency. Allows for better visualization if many points overlay one another]. Defaults to 1.0.

		colors ([dict], optional): [Dictionary with unique population IDs as keys and hex color codes as values. If colors=None, a diverging color palette automatically gets applied.]. Defaults to None.

		palette (str, optional): [Set the automatically generated color palette if colors=None. See matplotlib.colors documentation]. Defaults to "Set1".

		legend (boolean, optional): [If True, a legend is included]. Defaults to True.

		legend_inside (boolean, optional): [If True, the legend is located inside the plot]. Defaults to False.

		legend_loc (str, optional): [Set the location of the legend. If some of your points get covered with the legend you can change its location]. Defaults to "upper left".

		marker (str, optional): [Set the marker shape. See matplotlib.markers documentation]. Defaults to 'o' (a circle).

		markersize (int, optional): [Set size of the markers]. Defaults to 6.

		markeredgecolor (str, optional): [Set the color of the marker edge]. Defaults to 'k' (black). See matplotlib.pyplot.plot documentation.

		markeredgewidth (float, optional): [Set the width of the marker edge]. Defaults to 0.5.

		labelspacing (float, optional): [The vertical space between the legend entries, in font-size units]. Defaults to 0.5.

		columnspacing (float, optional): [The spacing between columns, in font-size units]. Defaults to 2.0.

		title (str or None, optional): [The legend's title]. Defaults to None.

		title_fontsize (int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, optional): [The font size of the legend's title]. Defaults to None.

		markerfirst (boolean, optional): [If True, legend marker is placed to the left of the legend label. If False, legend marker is placed to the right of the legend label]. Defaults to True.

		markerscale (float, optional): [The relative size of legend markers compared with the originally drawn ones]. Defaults to 1.0.

		ncol (int, optional): [The number of columns that the legend has]. Defaults to 1.

		bbox_to_anchor (tuple(float, float) or tuple(float, float, float, float), optional): [Box that is used to position the legend in conjuction with legend_loc. If a 4-tuple bbox is given, then it specifies the bbox(x, y, width, height) that the legend is placed in]. Defaults to (1.0, 1.0).

		borderaxespad (float, optional): [The pad between the axes and legend border, in font-size units]. Defaults to 0.5.

		legend_edgecolor (float, optional): [The legend's background patch edge color. If "inherit", use take rcParams["axes.edgecolor"]]. Defaults to "black".

		facecolor (str, optional): [The legend's background color. If "inherit", use rcParams["axes.facecolor"]]. Defaults to "white".

		framealpha (float, optional): [The alpha transparency of the legend's background. If shadow is activated and framealpha is None, the default value is ignored]. Defaults to 0.8.

		shadow (boolean, optional): [Whether to draw a shadow behind the legend]. Defaults to False.
	"""
	dimreduction_settings = {
		"axis1": 1, 
		"axis2": 2,
		"axis3": 3, 
		"figwidth": 6, 
		"figheight": 6, 
		"alpha": 1.0, 
		"colors": None, 
		"palette": "Set1", 
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
		"ncol": 1, 
		"bbox_to_anchor": (1.0, 1.0), 
		"borderaxespad": 0.5, 
		"legend_edgecolor": "black", 
		"facecolor": "white", 
		"framealpha": 0.8, 
		"shadow": False
	}

	return dimreduction_settings

def pca_cumvar_default_settings():
	"""[Default settings for plotting cumulative variance of PCA]

	Returns:
		[dict]: [matplotlib.pyplot arguments as keys with the corresponding values]
	"""
	pca_cumvar_settings = {
		"text_size": 14, 
		"style": "white", 
		"figwidth": 6, 
		"figheight": 6, 
		"linecolor": "blue",
		"linewidth": 3,
		"xintercept_width": 3, 
		"xintercept_color": "r", 
		"xintercept_style": "--"
	}

	return pca_cumvar_settings
	



