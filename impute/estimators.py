# Standard library imports
import sys
from pathlib import Path
from timeit import default_timer
from typing import Optional, Union, List, Dict, Tuple, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
import scipy.linalg
import toyplot.pdf
import toyplot as tp
import toytree as tt

# Scikit-learn imports
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier


# Custom imports
from read_input.read_input import GenotypeData
from impute.impute import Impute
from impute.neural_network_imputers import VAE, UBP

from utils import misc
from utils.misc import get_processor_name
from utils.misc import isnotebook
from utils.misc import timer

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        intelex = True
    except ImportError:
        print(
            "Warning: Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False


class ImputeKNN:
    """Does K-Nearest Neighbors Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``, a grid search is not performed. Defaults to None.

        grid_iter (int, optional): Number of iterations for grid search. Defaults to 50.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        validation_only (float or None, optional): Validates the imputation without doing a grid search. The validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None. Defaults to 0.4.

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead. Defaults to False.

        population_size (int, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation. Defaults to 10.

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.1.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        n_neighbors (int, optional): Number of neighbors to use by default for K-Nearest Neighbors queries. Defaults to 5.

        weights (str, optional): Weight function used in prediction. Possible values: 'Uniform': Uniform weights with all points in each neighborhood weighted equally; 'distance': Weight points by the inverse of their distance, in this case closer neighbors of a query point will have  a greater influence than neighbors that are further away; 'callable': A user-defined function that accepts an array of distances and returns an array of the same shape containing the weights. Defaults to "distance".

        algorithm (str, optional): Algorithm used to compute the nearest neighbors. Possible values: 'ball_tree', 'kd_tree', 'brute', 'auto'. Defaults to "auto".

        leaf_size (int, optional): Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. Defaults to 30.

        p (int, optional): Power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance (l1), and if p=2 it is equivalent to using euclidean distance (l2). For arbitrary p, minkowski_distance (l_p) is used. Defaults to 2.

        metric (str, optional): The distance metric to use for the tree. The default metric is minkowski, and with p=2 this is equivalent to the standard Euclidean metric. See the documentation of sklearn.DistanceMetric for a list of available metrics. If metric is 'precomputed', X is assumed to be a distance matrix and must be square during fit. Defaults to "minkowski".

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        grid_iter: int = 50,
        cv: int = 5,
        validation_only: float = 0.4,
        ga: bool = False,
        population_size: int = 10,
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.1,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        column_subset: Union[int, float] = 0.1,
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        n_neighbors: int = 5,
        weights: str = "distance",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "most_frequent",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()

        self.clf_type = "classifier"
        self.clf = KNeighborsClassifier

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeRandomForest:
    """Does Random Forest or Extra Trees Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV), the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``, a grid search is not performed. Defaults to None.

        grid_iter (int, optional): Number of iterations for grid search. Defaults to 50.

        cv (int, optional): Number of folds for cross-validation during grid search. Defaults to 5.

        validation_only (float or None, optional): Validates the imputation without doing a grid search. The validation method randomly replaces between 15% and 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None. Defaults to 0.4.

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead. Defaults to False.

        population_size (int, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation. Defaults to 10.

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.1.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        extra_trees (bool, optional): Whether to use ExtraTreesClassifier (If True) instead of RandomForestClassifier (If False). ExtraTreesClassifier is faster, but is not supported by the scikit-learn-intelex patch, whereas RandomForestClassifier is. If using an Intel CPU, the optimizations provided by the scikit-learn-intelex patch might make setting ``extratrees=False`` worthwhile. If you are not using an Intel CPU, the scikit-learn-intelex library is not supported and ExtraTreesClassifier will be faster with similar performance. NOTE: If using scikit-learn-intelex, ``criterion`` must be set to "gini" and ``oob_score`` to False, as those parameters are not currently supported herein. Defaults to True.

        n_estimators (int, optional): The number of trees in the forest. Increasing this value can improve the fit, but at the cost of compute time and resources. Defaults to 100.

        criterion (str, optional): The function to measure the quality of a split. Supported values are "gini" for the Gini impurity and "entropy" for the information gain. Defaults to "gini".

        max_depth (int, optional): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. Defaults to None.

        min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. If value is an integer, then considers min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples), rounded up to the nearest integer, are the minimum number of samples for each split. Defaults to 2.

        min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, then ``min_samples_leaf`` is the minimum number. If value is floating point, then ``min_samples_leaf`` is a fraction and ``int(min_samples_leaf * n_samples)`` is the minimum number of samples for each node. Defaults to 1.

        min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. Defaults to 0.0.

        max_features (str, int, float, or None, optional): The number of features to consider when looking for the best split. If int, then consider "max_features" features at each split. If float, then "max_features" is a fraction and ``int(max_features * n_samples)`` features are considered at each split. If "auto", then ``max_features=sqrt(n_features)``. If "sqrt", then ``max_features=sqrt(n_features)``. If "log2", then ``max_features=log2(n_features)``. If None, then ``max_features=n_features``. Defaults to "auto".

        max_leaf_nodes (int or None, optional): Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. Defaults to None.

        min_impurity_decrease (float, optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. See ``sklearn.ensemble.ExtraTreesClassifier`` documentation for more information. Defaults to 0.0.

        bootstrap (bool, optional): Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree. Defaults to False.

        oob_score (bool, optional): Whether to use out-of-bag samples to estimate the generalization score. Only available if ``bootstrap=True``. Defaults to False.

        max_samples (int or float, optional): If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draws ``X.shape[0] samples``. if int, then draws ``max_samples`` samples. If float, then draws ``int(max_samples * X.shape[0] samples)`` with ``max_samples`` in the interval (0, 1). Defaults to None.

        clf_random_state (int or None, optional): Controls three sources of randomness for ``sklearn.ensemble.ExtraTreesClassifier``: The bootstrapping of the samples used when building trees (if ``bootstrap=True``), the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``), and the draw of the splits for each of the ``max_features``. If None, then uses a different random seed each time the imputation is run. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population based on a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        grid_iter: int = 50,
        cv: int = 5,
        validation_only: Optional[float] = 0.4,
        ga: bool = False,
        population_size: int = 10,
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.1,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        column_subset: Union[int, float] = 0.1,
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        extratrees: bool = True,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[str, int, float]] = "auto",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        max_samples: Optional[Union[int, float]] = None,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()

        self.extratrees = kwargs.pop("extratrees")

        if self.extratrees:
            self.clf = ExtraTreesClassifier

        elif intelex and not self.extratrees:
            self.clf = RandomForestClassifier

            if kwargs["criterion"] != "gini":
                raise ValueError(
                    "criterion must be set to 'gini' if using the RandomForestClassifier with scikit-learn-intelex"
                )
            if kwargs["oob_score"]:
                raise ValueError(
                    "oob_score must be set to False if using the RandomForestClassifier with scikit-learn-intelex"
                )
        else:
            self.clf = RandomForestClassifier

        self.clf_type = "classifier"

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeGradientBoosting:
    """Does Gradient Boosting Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV), the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``, a grid search is not performed. Defaults to None.

        grid_iter (int, optional): Number of iterations for randomized grid search. Defaults to 50.

        cv (int, optional): Number of folds for cross-validation during randomized grid search. Defaults to 5.

        validation_only (float or None, optional): Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None. Defaults to 0.4.

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead. Defaults to False.

        population_size (int, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation. Defaults to 10.

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.1.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        n_estimators (int, optional): The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance but also increases compute time and required resources. Defaults to 100.

        loss (str, optional): The loss function to be optimized. "deviance" refers to deviance (=logistic regression) for classification with probabilistic outputs. For loss "exponential" gradient boosting recovers the AdaBoost algorithm. Defaults to "deviance".

        learning_rate (float, optional): Learning rate shrinks the contribution of each tree by ``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``. Defaults to 0.1.

        subsample (float, optional): The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. ``subsample`` interacts with the parameter ``n_estimators``. Choosing ``subsample < 1.0`` leads to a reduction of variance and an increase in bias. Defaults to 1.0.

        criterion (str, optional): The function to measure the quality of a split. Supported criteria are "friedman_mse" for the mean squared error with improvement score by Friedman and "mse" for mean squared error. The default value of "friedman_mse" is generally the best as it can provide a better approximation in some cases. Defaults to "friedman_mse".

        min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. If value is an integer, then consider ``min_samples_split`` as the minimum number. If value is a floating point, then min_samples_split is a fraction and ``(min_samples_split * n_samples)`` is rounded up to the nearest integer and used as the number of samples per split. Defaults to 2.

        min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, consider ``min_samples_leaf`` as the minimum number. If value is a floating point, then ``min_samples_leaf`` is a fraction and ``(min_samples_leaf * n_samples)`` rounded up to the nearest integer is the minimum number of samples per node. Defaults to 1.

        min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when ``sample_weight`` is not provided. Defaults to 0.0.

        max_depth (int, optional): The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables. Defaults to 3.

        min_impurity_decrease (float, optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Defaults to 0.0. See ``sklearn.ensemble.GradientBoostingClassifier`` documentation for more information. Defaults to 0.0.

        max_features (int, float, str, or None, optional): The number of features to consider when looking for the best split. If value is an integer, then consider ``max_features`` features at each split. If value is a floating point, then ``max_features`` is a fraction and ``(max_features * n_features)`` is rounded to the nearest integer and considered as the number of features per split. If "auto", then ``max_features=sqrt(n_features)``. If "sqrt", then ``max_features=sqrt(n_features)``. If "log2", then ``max_features=log2(n_features)``. If None, then ``max_features=n_features``. Defaults to None.

        max_leaf_nodes (int or None, optional): Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then uses an unlimited number of leaf nodes. Defaults to None.

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split. Pass an int for reproducible output across multiple function calls. If None, then uses a different random seed for each function call. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        grid_iter: int = 50,
        cv: int = 5,
        validation_only: Optional[float] = 0.4,
        ga: bool = False,
        population_size: int = 10,
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.1,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        column_subset: Union[int, float] = 0.1,
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        n_estimators: int = 100,
        loss: str = "deviance",
        learning_rate: float = 0.1,
        subsample: Union[int, float] = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: Optional[int] = 3,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[str, int, float]] = None,
        max_leaf_nodes: Optional[int] = None,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()

        self.clf_type = "classifier"
        self.clf = GradientBoostingClassifier

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeBayesianRidge:
    """NOTE: This is a regressor estimator and is only intended for testing purposes, as it is faster than the classifiers. Does Bayesian Ridge Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

         gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``, a grid search is not performed. Defaults to None.

        grid_iter (int, optional): Number of iterations for randomized grid search. Defaults to 50.

        cv (int, optional): Number of folds for cross-validation during randomized grid search. Defaults to 5.

        validation_only (float or None, optional): Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``int(n_features * validation_only)`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None. Defaults to 0.4.

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead. Defaults to False.

        population_size (int, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation. Defaults to 10.

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.1.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors. Defaults to 1.

        n_iter (int, optional): Maximum number of iterations. Should be greater than or equal to 1. Defaults to 300.

        clf_tol (float, optional): Stop the algorithm if w has converged. Defaults to 1e-3.

        alpha_1 (float, optional): Hyper-parameter: shape parameter for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.

        alpha_2 (float, optional): Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.

        lambda_1 (float, optional): Hyper-parameter: shape parameter for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.

        lambda_2 (float, optional): Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.

        alpha_init (float or None, optional): Initial value for alpha (precision of the noise). If None, ``alpha_init`` is ``1/Var(y)``. Defaults to None.

        lambda_init (float or None, optional): Initial value for lambda (precision of the weights). If None, ``lambda_init`` is 1. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        grid_iter: int = 50,
        cv: int = 5,
        validation_only: Optional[float] = 0.4,
        ga: bool = False,
        population_size: int = 10,
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.1,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "neg_root_mean_squared_error",
        column_subset: Union[int, float] = 0.1,
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        n_iter: int = 300,
        clf_tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        alpha_init: Optional[float] = None,
        lambda_init: Optional[float] = None,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = 10,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()
        kwargs["normalize"] = True
        kwargs["sample_posterior"] = False

        self.clf_type = "regressor"
        self.clf = BayesianRidge

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeXGBoost:
    """Does XGBoost (Extreme Gradient Boosting) Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process. The grid searches are not compatible with XGBoost, but validation scores can still be calculated without a grid search. In addition, ImputeLightGBM is a similar algorithm and is compatible with grid searches, so use ImputeLightGBM if you want a grid search.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        cv (int, optional): Number of folds for cross-validation during randomized grid search. Defaults to 5.

        validation_only (float or None, optional): Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``int(n_features * validation_only)`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None. Defaults to 0.4.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        n_estimators (int, optional): The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage. Defaults to 100.

        max_depth (int, optional): Maximum tree depth for base learners. Defaults to 3.

        learning_rate (float, optional): Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources. Defaults to 0.1.

        booster (str, optional): Specify which booster to use. Possible values include "gbtree", "gblinear", and "dart". Defaults to "gbtree".

        gamma (float, optional): Minimum loss reduction required to make a further partition on a leaf node of the tree. Defaults to 0.0.

        min_child_weight (float, optional): Minimum sum of instance weight(hessian) needed in a child. Defaults to 1.0.

        max_delta_step (float, optional): Maximum delta step we allow each tree's weight estimation to be. Defaults to 0.0.

        subsample (float, optional): Subsample ratio of the training instance. Defaults to 1.0.

        colsample_bytree (float, optional): Subsample ratio of columns when constructing each tree. Defaults to 1.0.

        reg_lambda (float, optional): L2 regularization term on weights (xgb's lambda parameter). Defaults to 1.0.

        reg_alpha (float, optional): L1 regularization term on weights (xgb's alpha parameter). Defaults to 1.0.

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Controls three sources of randomness for ``sklearn.ensemble.ExtraTreesClassifier``: The bootstrapping of the samples used when building trees (if ``bootstrap=True``), the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``), and the draw of the splits for each of the ``max_features``. If None, then uses a different random seed each time the imputation is run. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        cv: int = 5,
        validation_only: Optional[float] = 0.4,
        scoring_metric: str = "accuracy",
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        booster: str = "gbtree",
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        max_delta_step: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_nearest_features: Optional[int] = 10,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        # Get local variables into dictionary object
        kwargs = locals()
        kwargs["gridparams"] = None
        # kwargs["num_class"] = 3
        # kwargs["use_label_encoder"] = False

        self.clf_type = "classifier"
        self.clf = xgb.XGBClassifier
        kwargs["verbosity"] = verbose

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputeLightGBM:
    """Does LightGBM (Light Gradient Boosting) Iterative imputation of missing data. LightGBM is an alternative to XGBoost that is around 7X faster and uses less memory, while still maintaining high accuracy. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``, a grid search is not performed. Defaults to None.

        grid_iter (int, optional): Number of iterations for randomized grid search. Defaults to 50.

        cv (int, optional): Number of folds for cross-validation during randomized grid search. Defaults to 5.

        validation_only (float or None, optional): Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``int(n_features * validation_only)`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None. Defaults to 0.4.

        ga (bool, optional): Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead. Defaults to False.

        population_size (int, optional): For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation. Defaults to 10.

        tournament_size (int, optional): For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation. Defaults to 3.

        elitism (bool, optional): For genetic algorithm grid search: If True takes the tournament_size best solution to the next generation. See GASearchCV documentation. Defaults to True.

        crossover_probability (float, optional): For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation. Defaults to 0.8.

        mutation_probability (float, optional): For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation. Defaults to 0.1.

        ga_algorithm (str, optional): For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        n_estimators (int, optional): The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage. Defaults to 100.

        max_depth (int, optional): Maximum tree depth for base learners. Defaults to 3.

        learning_rate (float, optional): Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources. Defaults to 0.1.

        boosting_type (str, optional): Possible values: "gbdt", traditional Gradient Boosting Decision Tree. "dart", Dropouts meet Multiple Additive Regression Trees. "goss", Gradient-based One-Side Sampling. The "rf" option is not currently supported. Defaults to "gbdt".

        num_leaves (int, optional): Maximum tree leaves for base learners. Defaults to 31.

        subsample_for_bin (int, optional): Number of samples for constructing bins. Defaults to 200000.

        min_split_gain (float, optional): Minimum loss reduction required to make a further partition on a leaf node of the tree. Defaults to 0.0.

        min_child_weight (float, optional): Minimum sum of instance weight (hessian) needed in a child (leaf). Defaults to 1e-3.

        min_child_samples (int, optional): Minimum number of data needed in a child (leaf). Defaults to 20.

        subsample (float, optional): Subsample ratio of the training instance. Defaults to 1.0.

        subsample_freq (int, optional): Frequency of subsample, <=0 means no enable. Defaults to 0.

        colsample_bytree (float, optional): Subsample ratio of columns when constructing each tree. Defaults to 1.0.

        reg_lambda (float, optional): L2 regularization term on weights. Defaults to 0.

        reg_alpha (float, optional): L1 regularization term on weights. Defaults to 0.

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Random number seed. If int, this number is used to seed the C++ code. If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code. If None, default seeds in C++ code are used. Defaults to None.

        silent (bool, optional): Whether to print messages while running boosting. Defaults to True.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data: Any,
        *,
        prefix: str = "output",
        gridparams: Optional[Dict[str, Any]] = None,
        grid_iter: int = 50,
        cv: int = 5,
        validation_only: Optional[float] = 0.4,
        ga: bool = False,
        population_size: int = 10,
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.1,
        ga_algorithm: str = "eaMuPlusLambda",
        early_stop_gen: int = 5,
        scoring_metric: str = "accuracy",
        column_subset: Union[int, float] = 0.1,
        chunk_size: Union[int, float] = 1.0,
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        n_jobs: int = 1,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        subsample_for_bin: int = 200000,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 0.0,
        reg_alpha: float = 0.0,
        clf_random_state: Optional[Union[int, np.random.RandomState]] = None,
        silent: bool = True,
        n_nearest_features: Optional[int] = 10,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "populations",
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ) -> None:

        # Get local variables into dictionary object
        kwargs = locals()

        if kwargs["boosting_type"] == "rf":
            raise ValueError("boosting_type 'rf' is not supported!")

        self.clf_type = "classifier"
        self.clf = lgbm.LGBMClassifier

        imputer = Impute(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = imputer.fit_predict(
            genotype_data.genotypes_df
        )

        imputer.write_imputed(self.imputed)


class ImputePhylo(GenotypeData):
    """Impute missing data using a phylogenetic tree to inform the imputation.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData object. If not None, some or all of the below options are not required. If None, all the below options are required. Defaults to None.

        alnfile (str or None, optional): Path to PHYLIP or STRUCTURE-formatted file to impute. Defaults to None.

        filetype (str or None, optional): Filetype for the input alignment. Valid options include: "phylip", "structure1row", "structure1rowPopID", "structure2row", "structure2rowPopId". Not required if ``genotype_data`` is defined. Defaults to "phylip".

        popmapfile (str or None, optional): Path to population map file. Required if filetype is "phylip", "structure1row", or "structure2row". If filetype is "structure1rowPopID" or "structure2rowPopID", then the population IDs must be the second column of the STRUCTURE file. Not required if ``genotype_data`` is defined. Defaults to None.

        treefile (str or None, optional): Path to Newick-formatted phylogenetic tree file. Not required if ``genotype_data`` is defined with the ``guidetree`` option. Defaults to None.

        qmatrix_iqtree (str or None, optional): Path to *.iqtree file containing Rate Matrix Q table. If specified, ``ImputePhylo`` will read the Q matrix from the IQ-TREE output file. Cannot be used in conjunction with ``qmatrix`` argument. Not required if the ``qmatrix`` or ``qmatrix_iqtree`` options were used with the ``GenotypeData`` object. Defaults to None.

        qmatrix (str or None, optional): Path to file containing only a Rate Matrix Q table. Not required if ``genotype_data`` is defined with the qmatrix or qmatrix_iqtree option. Defaults to None.

        str_encodings (Dict[str, int], optional): Integer encodings used in STRUCTURE-formatted file. Should be a dictionary with keys=nucleotides and values=integer encodings. The missing data encoding should also be included. Argument is ignored if using a PHYLIP-formatted file. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}

        prefix (str, optional): Prefix to use with output files.

        save_plots (bool, optional): Whether to save PDF files with genotype imputations for each site to disk. It makes one PDF file per locus, so if you have a lot of loci it will make a lot of PDF files. Defaults to False.

        write_output (bool, optional): Whether to save the imputed data to disk. Defaults to True.

        disable_progressbar (bool, optional): Whether to disable the progress bar during the imputation. Defaults to False.

        kwargs (Dict[str, Any] or None, optional): Additional keyword arguments intended for internal purposes only. Possible arguments: {"column_subset": List[int] or numpy.ndarray[int]}; Subset SNPs by a list of indices. Defauls to None.
    """

    def __init__(
        self,
        *,
        genotype_data: Optional[Any] = None,
        alnfile: Optional[str] = None,
        filetype: Optional[str] = None,
        popmapfile: Optional[str] = None,
        treefile: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        prefix: str = "output",
        save_plots: bool = False,
        write_output: bool = True,
        disable_progressbar: bool = False,
        **kwargs: Optional[Any],
    ) -> None:
        super().__init__()

        self.alnfile = alnfile
        self.filetype = filetype
        self.popmapfile = popmapfile
        self.treefile = treefile
        self.qmatrix_iqtree = qmatrix_iqtree
        self.qmatrix = qmatrix
        self.str_encodings = str_encodings
        self.prefix = prefix
        self.save_plots = save_plots
        self.disable_progressbar = disable_progressbar
        self.column_subset = kwargs.get("column_subset", None)

        self.valid_sites = None
        self.valid_sites_count = None

        self.validate_arguments(genotype_data)
        data, tree, q = self.parse_arguments(genotype_data)

        self.imputed = self.impute_phylo(tree, data, q)

        if write_output:
            outfile = f"{prefix}_imputed_012.csv"
            self.imputed.to_csv(outfile, header=False, index=False)

    def nbiallelic(self) -> int:
        """Get the number of remaining bi-allelic sites after imputation.

        Returns:
            int: Number of bi-allelic sites remaining after imputation.
        """
        return len(self.imputed.columns)

    def parse_arguments(
        self, genotype_data: Any
    ) -> Tuple[Dict[str, List[Union[int, str]]], tt.tree, pd.DataFrame]:
        """Determine which arguments were specified and set appropriate values.

        Args:
            genotype_data (GenotypeData object): Initialized GenotypeData object.

        Returns:
            Dict[str, List[Union[int, str]]]: GenotypeData.snpsdict object. If genotype_data is not None, then this value gets set from the GenotypeData.snpsdict object. If alnfile is not None, then the alignment file gets read and the snpsdict object gets set from the alnfile.

            toytree.tree: Input phylogeny, either read from GenotypeData object or supplied with treefile.

            pandas.DataFrame: Q Rate Matrix, either from IQ-TREE file or from its own supplied file.
        """
        if genotype_data is not None:
            data = genotype_data.snpsdict
            self.filetype = genotype_data.filetype

        elif self.alnfile is not None:
            self.parse_filetype(self.filetype, self.popmapfile)

        if genotype_data.tree is not None and self.treefile is None:
            tree = genotype_data.tree

        elif genotype_data.tree is not None and self.treefile is not None:
            print(
                "WARNING: Both genotype_data.tree and treefile are defined; using local definition"
            )
            tree = self.read_tree(self.treefile)

        elif genotype_data.tree is None and self.treefile is not None:
            tree = self.read_tree(self.treefile)

        if (
            genotype_data.q is not None
            and self.qmatrix is None
            and self.qmatrix_iqtree is None
        ):
            q = genotype_data.q

        elif genotype_data.q is None:
            if self.qmatrix is not None:
                q = self.q_from_file(self.qmatrix)
            elif self.qmatrix_iqtree is not None:
                q = self.q_from_iqtree(self.qmatrix_iqtree)

        elif genotype_data.q is not None:
            if self.qmatrix is not None:
                print(
                    "WARNING: Both genotype_data.q and qmatrix are defined; "
                    "using local definition"
                )
                q = self.q_from_file(self.qmatrix)
            if self.qmatrix_iqtree is not None:
                print(
                    "WARNING: Both genotype_data.q and qmatrix are defined; "
                    "using local definition"
                )
                q = self.q_from_iqtree(self.qmatrix_iqtree)

        return data, tree, q

    def validate_arguments(self, genotype_data: Any) -> None:
        """Validate that the correct arguments were supplied.

        Args:
            genotype_data (GenotypeData object): Input GenotypeData object.

        Raises:
            TypeError: Cannot define both genotype_data and alnfile.
            TypeError: Must define either genotype_data or phylipfile.
            TypeError: Must define either genotype_data.tree or treefile.
            TypeError: filetype must be defined if genotype_data is None.
            TypeError: Q rate matrix must be defined.
            TypeError: qmatrix and qmatrix_iqtree cannot both be defined.
        """
        if genotype_data is not None and self.alnfile is not None:
            raise TypeError("genotype_data and alnfile cannot both be defined")

        if genotype_data is None and self.alnfile is None:
            raise TypeError("Either genotype_data or phylipfle must be defined")

        if genotype_data.tree is None and self.treefile is None:
            raise TypeError(
                "Either genotype_data.tree or treefile must be defined"
            )

        if genotype_data is None and self.filetype is None:
            raise TypeError("filetype must be defined if genotype_data is None")

        if (
            genotype_data is None
            and self.qmatrix is None
            and self.qmatrix_iqtree is None
        ):
            raise TypeError(
                "q matrix must be defined in either genotype_data, "
                "qmatrix_iqtree, or qmatrix"
            )

        if self.qmatrix is not None and self.qmatrix_iqtree is not None:
            raise TypeError("qmatrix and qmatrix_iqtree cannot both be defined")

    def print_q(self, q: pd.DataFrame) -> None:
        """Print Rate Matrix Q.

        Args:
            q (pandas.DataFrame): Rate Matrix Q.
        """
        print("Rate matrix Q:")
        print("\tA\tC\tG\tT\t")
        for nuc1 in ["A", "C", "G", "T"]:
            print(nuc1, end="\t")
            for nuc2 in ["A", "C", "G", "T"]:
                print(q[nuc1][nuc2], end="\t")
            print("")

    def is_int(self, val: Union[str, int]) -> bool:
        """Check if value is integer.

        Args:
            val (int or str): Value to check.

        Returns:
            bool: True if integer, False if string.
        """
        try:
            num = int(val)
        except ValueError:
            return False
        return True

    def impute_phylo(
        self,
        tree: tt.tree,
        genotypes: Dict[str, List[Union[str, int]]],
        Q: pd.DataFrame,
    ) -> pd.DataFrame:
        """Imputes genotype values with a guide tree.

        Imputes genotype values by using a provided guide
        tree to inform the imputation, assuming maximum parsimony.

        Process Outline:
            For each SNP:
            1) if site_rates, get site-transformated Q matrix.

            2) Postorder traversal of tree to compute ancestral
            state likelihoods for internal nodes (tips -> root).
            If exclude_N==True, then ignore N tips for this step.

            3) Preorder traversal of tree to populate missing genotypes
            with the maximum likelihood state (root -> tips).

        Args:
            tree (toytree.tree object): Input tree.

            genotypes (Dict[str, List[Union[str, int]]]): Dictionary with key=sampleids, value=sequences.

            Q (pandas.DataFrame): Rate Matrix Q from .iqtree or separate file.

        Returns:
            pandas.DataFrame: Imputed genotypes.

        Raises:
            IndexError: If index does not exist when trying to read genotypes.
            AssertionError: Sites must have same lengths.
            AssertionError: Missing data still found after imputation.
        """
        try:
            if list(genotypes.values())[0][0][1] == "/":
                genotypes = self.str2iupac(genotypes, self.str_encodings)
        except IndexError:
            if self.is_int(list(genotypes.values())[0][0][0]):
                raise

        if self.column_subset is not None:
            if isinstance(self.column_subset, np.ndarray):
                self.column_subset = self.column_subset.tolist()

            genotypes = {
                k: [v[i] for i in self.column_subset]
                for k, v in genotypes.items()
            }

        # For each SNP:
        nsites = list(set([len(v) for v in genotypes.values()]))
        assert len(nsites) == 1, "Some sites have different lengths!"

        outdir = f"{self.prefix}_imputation_plots"

        if self.save_plots:
            Path(outdir).mkdir(parents=True, exist_ok=True)

        for snp_index in progressbar(
            range(nsites[0]),
            desc="Feature Progress: ",
            leave=True,
            disable=self.disable_progressbar,
        ):
            node_lik = dict()

            # LATER: Need to get site rates
            rate = 1.0

            site_Q = Q.copy(deep=True) * rate

            # calculate state likelihoods for internal nodes
            for node in tree.treenode.traverse("postorder"):
                if node.is_leaf():
                    continue

                if node.idx not in node_lik:
                    node_lik[node.idx] = None

                for child in node.get_leaves():
                    # get branch length to child
                    # bl = child.edge.length
                    # get transition probs
                    pt = self.transition_probs(site_Q, child.dist)
                    if child.is_leaf():
                        if child.name in genotypes:
                            # get genotype
                            sum = None

                            for allele in self.get_iupac_full(
                                genotypes[child.name][snp_index]
                            ):
                                if sum is None:
                                    sum = list(pt[allele])
                                else:
                                    sum = [
                                        sum[i] + val
                                        for i, val in enumerate(
                                            list(pt[allele])
                                        )
                                    ]

                            if node_lik[node.idx] is None:
                                node_lik[node.idx] = sum

                            else:
                                node_lik[node.idx] = [
                                    sum[i] * val
                                    for i, val in enumerate(node_lik[node.idx])
                                ]
                        else:
                            # raise error
                            sys.exit(
                                f"Error: Taxon {child.name} not found in "
                                f"genotypes"
                            )

                    else:
                        l = self.get_internal_lik(pt, node_lik[child.idx])
                        if node_lik[node.idx] is None:
                            node_lik[node.idx] = l

                        else:
                            node_lik[node.idx] = [
                                l[i] * val
                                for i, val in enumerate(node_lik[node.idx])
                            ]

            # infer most likely states for tips with missing data
            # for each child node:
            bads = list()
            for samp in genotypes.keys():
                if genotypes[samp][snp_index].upper() == "N":
                    bads.append(samp)
                    # go backwards into tree until a node informed by
                    # actual data
                    # is found
                    # node = tree.search_nodes(name=samp)[0]
                    node = tree.idx_dict[
                        tree.get_mrca_idx_from_tip_labels(names=samp)
                    ]
                    dist = node.dist
                    node = node.up
                    imputed = None

                    while node and imputed is None:
                        if self.allMissing(
                            tree, node.idx, snp_index, genotypes
                        ):
                            dist += node.dist
                            node = node.up

                        else:
                            pt = self.transition_probs(site_Q, dist)
                            lik = self.get_internal_lik(pt, node_lik[node.idx])
                            maxpos = lik.index(max(lik))
                            if maxpos == 0:
                                imputed = "A"

                            elif maxpos == 1:
                                imputed = "C"

                            elif maxpos == 2:
                                imputed = "G"

                            else:
                                imputed = "T"

                    genotypes[samp][snp_index] = imputed

            if self.save_plots:
                self.draw_imputed_position(
                    tree,
                    bads,
                    genotypes,
                    snp_index,
                    f"{outdir}/{self.prefix}_pos{snp_index}.pdf",
                )

        df = pd.DataFrame.from_dict(genotypes, orient="index")

        # Make sure no missing data remains in the dataset
        assert (
            not df.isin([-9]).any().any()
        ), "Imputation failed...Missing values found in the imputed dataset"

        imp_snps, self.valid_sites, self.valid_sites_count = self.convert_012(
            df.to_numpy().tolist(), impute_mode=True
        )

        df_imp = pd.DataFrame.from_records(imp_snps)

        return df_imp

    def get_nuc_colors(self, nucs: List[str]) -> List[str]:
        """Get colors for each nucleotide when plotting.

        Args:
            nucs (List[str]): Nucleotides at current site.

        Returns:
            List[str]: Hex-code color values for each IUPAC nucleotide.
        """
        ret = list()
        for nuc in nucs:
            nuc = nuc.upper()
            if nuc == "A":
                ret.append("#0000FF")  # blue
            elif nuc == "C":
                ret.append("#FF0000")  # red
            elif nuc == "G":
                ret.append("#00FF00")  # green
            elif nuc == "T":
                ret.append("#FFFF00")  # yellow
            elif nuc == "R":
                ret.append("#0dbaa9")  # blue-green
            elif nuc == "Y":
                ret.append("#FFA500")  # orange
            elif nuc == "K":
                ret.append("#9acd32")  # yellow-green
            elif nuc == "M":
                ret.append("#800080")  # purple
            elif nuc == "S":
                ret.append("#964B00")
            elif nuc == "W":
                ret.append("#C0C0C0")
            else:
                ret.append("#000000")
        return ret

    def label_bads(
        self, tips: List[str], labels: List[str], bads: List[str]
    ) -> List[str]:
        """Insert asterisks around bad nucleotides.

        Args:
            tips (List[str]): Tip labels (sample IDs).
            labels (List[str]): List of nucleotides at current site.
            bads (List[str]): List of tips that have missing data at current site.

        Returns:
            List[str]: IUPAC Nucleotides with "*" inserted around tips that had missing data.
        """
        for i, t in enumerate(tips):
            if t in bads:
                labels[i] = "*" + str(labels[i]) + "*"
        return labels

    def draw_imputed_position(
        self,
        tree: tt.tree,
        bads: List[str],
        genotypes: Dict[str, List[str]],
        pos: int,
        out: str = "tree.pdf",
    ) -> None:
        """Draw nucleotides at phylogeny tip and saves to file on disk.

        Draws nucleotides as tip labels for the current SNP site. Imputed values have asterisk surrounding the nucleotide label. The tree is converted to a toyplot object and saved to file.

        Args:
            tree (toytree.tree): Input tree object.
            bads (List[str]): List of sampleIDs that have missing data at the current SNP site.
            genotypes (Dict[str, List[str]]): Genotypes at all SNP sites.
            pos (int): Current SNP index.
            out (str, optional): Output filename for toyplot object.
        """

        # print(tree.get_tip_labels())
        sizes = [8 if i in bads else 0 for i in tree.get_tip_labels()]
        colors = [genotypes[i][pos] for i in tree.get_tip_labels()]
        labels = colors

        labels = self.label_bads(tree.get_tip_labels(), labels, bads)

        colors = self.get_nuc_colors(colors)

        mystyle = {
            "edge_type": "p",
            "edge_style": {
                "stroke": tt.colors[0],
                "stroke-width": 1,
            },
            "tip_labels_align": True,
            "tip_labels_style": {"font-size": "5px"},
            "node_labels": False,
        }

        canvas, axes, mark = tree.draw(
            tip_labels_colors=colors,
            tip_labels=labels,
            width=400,
            height=600,
            **mystyle,
        )

        toyplot.pdf.render(canvas, out)

    def allMissing(
        self,
        tree: tt.tree,
        node_index: int,
        snp_index: int,
        genotypes: Dict[str, List[str]],
    ) -> bool:
        """Check if all descendants of a clade have missing data at SNP site.

        Args:
            tree (toytree.tree): Input guide tree object.

            node_index (int): Parent node to determine if all desendants have missing data.

            snp_index (int): Index of current SNP site.

            genotypes (Dict[str, List[str]]): Genotypes at all SNP sites.

        Returns:
            bool: True if all descendants have missing data, otherwise False.
        """
        for des in tree.get_tip_labels(idx=node_index):
            if genotypes[des][snp_index].upper() not in ["N", "-"]:
                return False
        return True

    def get_internal_lik(
        self, pt: pd.DataFrame, lik_arr: List[float]
    ) -> List[float]:
        """Get ancestral state likelihoods for internal nodes of the tree.

        Postorder traversal to calculate internal ancestral state likelihoods (tips -> root).

        Args:
            pt (pandas.DataFrame): Transition probabilities calculated from Rate Matrix Q.
            lik_arr (List[float]): Likelihoods for nodes or leaves.

        Returns:
            List[float]: Internal likelihoods.
        """
        ret = list()
        for i, val in enumerate(lik_arr):
            col = list(pt.iloc[:, i])
            sum = 0.0
            for v in col:
                sum += v * val
            ret.append(sum)
        return ret

    def transition_probs(self, Q: pd.DataFrame, t: float) -> pd.DataFrame:
        """Get transition probabilities for tree.

        Args:
            Q (pd.DataFrame): Rate Matrix Q.
            t (float): Tree distance of child.

        Returns:
            pd.DataFrame: Transition probabilities.
        """
        ret = Q.copy(deep=True)
        m = Q.to_numpy()
        pt = scipy.linalg.expm(m * t)
        ret[:] = pt
        return ret

    def str2iupac(
        self, genotypes: Dict[str, List[str]], str_encodings: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """Convert STRUCTURE-format encodings to IUPAC bases.

        Args:
            genotypes (Dict[str, List[str]]): Genotypes at all sites.
            str_encodings (Dict[str, int]): Dictionary that maps IUPAC bases (keys) to integer encodings (values).

        Returns:
            Dict[str, List[str]]: Genotypes converted to IUPAC format.
        """
        a = str_encodings["A"]
        c = str_encodings["C"]
        g = str_encodings["G"]
        t = str_encodings["T"]
        n = str_encodings["N"]
        nuc = {
            f"{a}/{a}": "A",
            f"{c}/{c}": "C",
            f"{g}/{g}": "G",
            f"{t}/{t}": "T",
            f"{n}/{n}": "N",
            f"{a}/{c}": "M",
            f"{c}/{a}": "M",
            f"{a}/{g}": "R",
            f"{g}/{a}": "R",
            f"{a}/{t}": "W",
            f"{t}/{a}": "W",
            f"{c}/{g}": "S",
            f"{g}/{c}": "S",
            f"{c}/{t}": "Y",
            f"{t}/{c}": "Y",
            f"{g}/{t}": "K",
            f"{t}/{g}": "K",
        }

        for k, v in genotypes.items():
            for i, gt in enumerate(v):
                v[i] = nuc[gt]

        return genotypes

    def get_iupac_full(self, char: str) -> List[str]:
        """Map nucleotide to list of expanded IUPAC encodings.

        Args:
            char (str): Current nucleotide.

        Returns:
            List[str]: List of nucleotides in ``char`` expanded IUPAC.
        """
        char = char.upper()
        iupac = {
            "A": ["A"],
            "G": ["G"],
            "C": ["C"],
            "T": ["T"],
            "N": ["A", "C", "T", "G"],
            "-": ["A", "C", "T", "G"],
            "R": ["A", "G"],
            "Y": ["C", "T"],
            "S": ["G", "C"],
            "W": ["A", "T"],
            "K": ["G", "T"],
            "M": ["A", "C"],
            "B": ["C", "G", "T"],
            "D": ["A", "G", "T"],
            "H": ["A", "C", "T"],
            "V": ["A", "C", "G"],
        }

        ret = iupac[char]
        return ret


class ImputeAlleleFreq(GenotypeData):
    """Impute missing data by global allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData instance. If ``genotype_data`` is not defined, then ``gt`` must be defined instead, and they cannot both be defined. Defaults to None.

        gt (List[int] or None, optional): List of 012-encoded genotypes to be imputed. Either ``gt`` or ``genotype_data`` must be defined, and they cannot both be defined. Defaults to None.

        by_populations (bool, optional): Whether or not to impute by population or globally. Defaults to False (global allele frequency).

        pops (List[Union[str, int]] or None, optional): Population IDs in the same order as the samples. If ``by_populations=True``, then either ``pops`` or ``genotype_data`` must be defined. If both are defined, the ``pops`` argument will take priority. Defaults to None.

        diploid (bool, optional): When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency. Defaults to True.

        default (int, optional): Value to set if no alleles sampled at a locus. Defaults to 0.

        missing (int, optional): Missing data value. Defaults to -9.

        prefix (str, optional): Prefix for writing output files. Defaults to "output".

        write_output (bool, optional): Whether to save imputed output to a file. If ``write_output`` is False, then just returns the imputed values as a pandas.DataFrame object. If ``write_output`` is True, then it saves the imputed data as a CSV file called ``<prefix>_imputed_012.csv``.

        output_format (str, optional): Format of output imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list. Defaults to "df".

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        **kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool}. "iterative_mode" determines whether ``ImputeAlleleFreq`` is being used as the initial imputer in ``IterativeImputer``.

    Raises:
        TypeError: genotype_data and gt cannot both be NoneType.
        TypeError: genotype_data and gt cannot both be provided.
        TypeError: Either pops or genotype_data must be defined if by_populations is True.
    """

    def __init__(
        self,
        *,
        genotype_data: Optional[Any] = None,
        gt: Optional[List[int]] = None,
        by_populations: bool = False,
        pops: Optional[List[Union[str, int]]] = None,
        diploid: bool = True,
        default: int = 0,
        missing: int = -9,
        prefix: str = "output",
        write_output: bool = True,
        output_format: str = "df",
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:

        super().__init__()

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            gt_list = genotype_data.genotypes_list
        elif gt is not None:
            gt_list = gt

        if by_populations:
            if pops is None and genotype_data is None:
                raise TypeError(
                    "When by_populations is True, either pops or genotype_data must be defined"
                )

            if genotype_data is not None and pops is not None:
                print(
                    "WARNING: Both pops and genotype_data are defined. Using populations from pops argument"
                )
                self.pops = pops

            elif genotype_data is not None and pops is None:
                self.pops = genotype_data.populations

            elif genotype_data is None and pops is not None:
                self.pops = pops

        else:
            if pops is not None:
                print(
                    "WARNING: by_populations is False but pops is defined. Setting pops to None"
                )

            self.pops = None

        self.diploid = diploid
        self.default = default
        self.missing = missing
        self.prefix = prefix
        self.output_format = output_format
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)

        self.imputed, self.valid_cols = self.fit_predict(gt_list)

        if write_output:
            self.write2file(self.imputed)

    def fit_predict(
        self, X: List[List[int]]
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]],
        List[int],
    ]:
        """Impute missing genotypes using allele frequencies.

        Impute using global or by_population allele frequencies. Missing alleles are primarily coded as negative; usually -9.

        Args:
            X (List[List[int]], numpy.ndarray, or pandas.DataFrame): 012-encoded genotypes obtained from the GenotypeData object.

        Returns:
            pandas.DataFrame, numpy.ndarray, or List[List[Union[int, float]]]: Imputed genotypes of same shape as data.

            List[int]: Column indexes that were retained.

        Raises:
            TypeError: X must be either 2D list, numpy.ndarray, or pandas.DataFrame.

            ValueError: Unknown output_format type specified.
        """
        if self.pops is not None and self.verbose:
            print("\nImputing by population allele frequencies...")
        elif self.pops is None and self.verbose:
            print("\nImputing by global allele frequency...")

        if isinstance(X, (list, np.ndarray)):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(
                f"X must be of type list(list(int)), numpy.ndarray, "
                f"or pandas.DataFrame, but got {type(X)}"
            )

        df.replace(self.missing, np.nan, inplace=True)

        data = pd.DataFrame()
        valid_cols = list()
        bad_cnt = 0
        if self.pops is not None:
            # Impute per-population mode.
            # Loop method is faster (by 2X) than no-loop transform.
            df["pops"] = self.pops
            for col in df.columns:
                try:
                    data[col] = df.groupby(["pops"], sort=False)[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0])
                    )

                    # If all populations contained at least one non-NaN value.
                    if col != "pops":
                        valid_cols.append(col)

                except IndexError as e:
                    if str(e).lower().startswith("single positional indexer"):
                        bad_cnt += 1
                        # Impute with global mode
                        data[col] = df[col].fillna(df[col].mode().iloc[0])
                    else:
                        raise

            if bad_cnt > 0:
                print(
                    f"Warning: {bad_cnt} columns were imputed with the global "
                    f"mode because some of the populations "
                    f"contained only missing data"
                )

            data.drop("pops", axis=1, inplace=True)
        else:
            # Impute global mode.
            # No-loop method was faster for global.
            data = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=1)

        if self.iterative_mode:
            data = data.astype(dtype="float32")
        else:
            data = data.astype(dtype="Int8")

        if self.verbose:
            print("Done!")

        if self.output_format == "df":
            return data, valid_cols

        elif self.output_format == "array":
            return data.to_numpy(), valid_cols

        elif self.output_format == "list":
            return data.values.tolist(), valid_cols

        else:
            raise ValueError("Unknown output_format type specified!")

    def write2file(
        self, X: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]
    ) -> None:
        """Write imputed data to file on disk.

        Args:
            X (pandas.DataFrame, numpy.ndarray, List[List[Union[int, float]]]): Imputed data to write to file.

        Raises:
            TypeError: If X is of unsupported type.
        """
        outfile = f"{self.prefix}_imputed_012.csv"

        if isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)
        else:
            raise TypeError(
                f"Could not write imputed data because it is of incorrect "
                f"type. Got {type(X)}"
            )

        df.to_csv(outfile, header=False, index=False)


class ImputeVAE:
    """Class to impute missing data using a Variational Autoencoder neural network model.

    Args:
        genotype_data (GenotypeData object or None): Input data initialized as GenotypeData object. If value is None, then uses ``gt`` to get the genotypes. Either ``genotype_data`` or ``gt`` must be defined. Defaults to None.

        gt (numpy.ndarray or None): Input genotypes directly as a numpy array. If this value is None, ``genotype_data`` must be supplied instead. Defaults to None.

        prefix (str): Prefix for output files. Defaults to "output".

        cv (int): Number of cross-validation replicates to perform. Only used if ``validation_only`` is not None. Defaults to 5.

        initial_strategy (str): Initial strategy to impute missing data with for validation. Possible options include: "populations", "most_frequent", and "phylogeny", where "populations" imputes by the mode of each population at each site, "most_frequent" imputes by the overall mode of each site, and "phylogeny" uses an input phylogeny to inform the imputation. "populations" requires a population map file as input in the GenotypeData object, and "phylogeny" requires an input phylogeny and Rate Matrix Q (also instantiated in the GenotypeData object). Defaults to "populations".

        validation_only (float or None): Proportion of sites to use for the validation. If ``validation_only`` is None, then does not perform validation. Defaults to 0.2.

        disable_progressbar (bool): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        train_epochs (int): Number of epochs to train the VAE model with. Defaults to 100.

        batch_size (int): Batch size to train the model with.

        recurrent_weight (float): Weight to apply to recurrent network. Defaults to 0.5.

        optimizer (str): Gradient descent optimizer to use. See tf.keras.optimizers for more info. Defaults to "adam".

        dropout_probability (float): Dropout rate for neurons in the network. Can adjust to reduce overfitting. Defaults to 0.2.

        hidden_activation (str): Activation function to use for hidden layers. See tf.keras.activations for more info. Defaults to "relu".

        output_activation (str): Activation function to use for output layer. See tf.keras.activations for more info. Defaults to "sigmoid".

        kernel_initializer (str): Initializer to use for initializing model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float): L1 regularization penalty to apply. Adjust if overfitting is occurring. Defaults to 0.

        l2_penalty (float): L2 regularization penalty to apply. Adjust if overfitting is occurring. Defaults to 0.
    """

    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        prefix="output",
        cv=5,
        initial_strategy="populations",
        validation_only=0.2,
        disable_progressbar=False,
        chunk_size=1.0,
        train_epochs=100,
        batch_size=32,
        recurrent_weight=0.5,
        optimizer="adam",
        dropout_probability=0.2,
        hidden_activation="relu",
        output_activation="sigmoid",
        kernel_initializer="glorot_normal",
        l1_penalty=0,
        l2_penalty=0,
    ):

        # Get local variables into dictionary object
        all_kwargs = locals()

        self.clf = VAE
        self.clf_type = "classifier"

        imp_kwargs = {
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        all_kwargs.update(imp_kwargs)

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            X = genotype_data.genotypes_nparray

        elif gt is not None:
            X = gt

        imputer = Impute(self.clf, self.clf_type, all_kwargs)

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self._imputed, self._best_params = imputer.fit_predict(df)

        imputer.write_imputed(self._imputed)

    @property
    def imputed(self):
        return self._imputed

    @property
    def best_params(self):
        return self._best_params


class ImputeUBP:
    """Class to impute missing data using unsupervised backpropagation neural network models.

    Args:
        genotype_data (GenotypeData object): Input GenotypeData object.

        gt (numpy.ndarray or None): Input genotypes directly as a numpy array. If this value is None, ``genotype_data`` must be supplied instead. Defaults to None.

        prefix (str, optional): Prefix for output files. Defaults to "output".

        cv (int): Number of cross-validation replicates to perform. Only used if ``validation_only`` is not None. Defaults to 5.

        initial_strategy (str, optional): Initial strategy to impute missing data with for validation. Possible options include: "populations", "most_frequent", and "phylogeny", where "populations" imputes by the mode of each population at each site, "most_frequent" imputes by the overall mode of each site, and "phylogeny" uses an input phylogeny to inform the imputation. "populations" requires a population map file as input in the GenotypeData object, and "phylogeny" requires an input phylogeny and Rate Matrix Q (also instantiated in the GenotypeData object). Defaults to "populations".

        validation_only (float or None, optional): Proportion of sites to use for the validation. If ``validation_only`` is None, then does not perform validation. Defaults to 0.2.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        batch_size (int, optional): Batch size per epoch to train the model with.

        n_components (int, optional): Number of components to use as the input data. Defaults to 3.

        early_stop_gen (int, optional): Early stopping criterion for epochs. Training will stop if the loss (error) does not decrease past the tolerance level ``tol`` for ``early_stop_gen`` epochs. Will save the optimal model and reload it once ``early_stop_gen`` has been reached. Defaults to 50.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting occurs. Defaults to 3.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", and "adagrad" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        max_epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        tol (float, optional): Tolerance level to use for the early stopping criterion. If the loss does not improve past the tolerance level after ``early_stop_gen`` epochs, then training will halt. Defaults to 1e-3.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional) L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.
    """
    nlpca=False
    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        prefix="output",
        cv=5,
        initial_strategy="populations",
        validation_only=0.3,
        write_output=True,
        disable_progressbar=False,
        chunk_size=1.0,
        batch_size=32,
        n_components=3,
        early_stop_gen=50,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.1,
        max_epochs=1000,
        tol=1e-3,
        weights_initializer="glorot_normal",
        l1_penalty=0.01,
        l2_penalty=0.01,
    ):

        # Get local variables into dictionary object
        settings = locals()
        settings["nlpca"]=self.nlpca

        self.clf = UBP
        self.clf_type = "classifier"
        if self.nlpca:
            self.clf.__name__ = "NLPCA"

        imp_kwargs = {
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        settings.update(imp_kwargs)

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            X = genotype_data.genotypes_nparray

        elif gt is not None:
            X = gt

        imputer = Impute(self.clf, self.clf_type, settings)

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self._imputed, self._best_params = imputer.fit_predict(df)

        if write_output:
            imputer.write_imputed(self._imputed)

    @property
    def imputed(self):
        return self._imputed

    @property
    def best_params(self):
        return self._best_params

class ImputeNLPCA(ImputeUBP):
    """Class to impute missing data using non-linear principal component analysis (NLPCA) neural network models.

    Args:
        genotype_data (GenotypeData object): Input GenotypeData object.

        gt (numpy.ndarray or None): Input genotypes directly as a numpy array. If this value is None, ``genotype_data`` must be supplied instead. Defaults to None.

        prefix (str, optional): Prefix for output files. Defaults to "output".

        cv (int): Number of cross-validation replicates to perform. Only used if ``validation_only`` is not None. Defaults to 5.

        initial_strategy (str, optional): Initial strategy to impute missing data with for validation. Possible options include: "populations", "most_frequent", and "phylogeny", where "populations" imputes by the mode of each population at each site, "most_frequent" imputes by the overall mode of each site, and "phylogeny" uses an input phylogeny to inform the imputation. "populations" requires a population map file as input in the GenotypeData object, and "phylogeny" requires an input phylogeny and Rate Matrix Q (also instantiated in the GenotypeData object). Defaults to "populations".

        validation_only (float or None, optional): Proportion of sites to use for the validation. If ``validation_only`` is None, then does not perform validation. Defaults to 0.2.

        disable_progressbar (bool, optional): Whether to disable the tqdm progress bar. Useful if you are doing the imputation on e.g. a high-performance computing cluster, where sometimes tqdm does not work correctly. If False, uses tqdm progress bar. If True, does not use tqdm. Defaults to False.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        batch_size (int, optional): Batch size per epoch to train the model with.

        n_components (int, optional): Number of components to use as the input data. Defaults to 3.

        early_stop_gen (int, optional): Early stopping criterion for epochs. Training will stop if the loss (error) does not decrease past the tolerance level ``tol`` for ``early_stop_gen`` epochs. Will save the optimal model and reload it once ``early_stop_gen`` has been reached. Defaults to 50.

        num_hidden_layers (int, optional): Number of hidden layers to use in the model. Adjust if overfitting occurs. Defaults to 3.

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", and "adagrad" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        max_epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        tol (float, optional): Tolerance level to use for the early stopping criterion. If the loss does not improve past the tolerance level after ``early_stop_gen`` epochs, then training will halt. Defaults to 1e-3.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional) L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.
    """
    nlpca=True
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    @property
    def imputed(self):
        return self._imputed

    @property
    def best_params(self):
        return self._best_params

class ImputeMF(GenotypeData):
    """Impute missing data using matrix factorization. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData instance. If ``genotype_data`` is not defined, then ``gt`` must be defined instead, and they cannot both be defined. Defaults to None.

        gt (List[int] or None, optional): List of 012-encoded genotypes to be imputed. Either ``gt`` or ``genotype_data`` must be defined, and they cannot both be defined. Defaults to None.

        by_populations (bool, optional): Whether or not to impute by population or globally. Defaults to False (global allele frequency).

        pops (List[Union[str, int]] or None, optional): Population IDs in the same order as the samples. If ``by_populations=True``, then either ``pops`` or ``genotype_data`` must be defined. If both are defined, the ``pops`` argument will take priority. Defaults to None.

        diploid (bool, optional): When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency. Defaults to True.

        default (int, optional): Value to set if no alleles sampled at a locus. Defaults to 0.

        missing (int, optional): Missing data value. Defaults to -9.

        prefix (str, optional): Prefix for writing output files. Defaults to "output".

        write_output (bool, optional): Whether to save imputed output to a file. If ``write_output`` is False, then just returns the imputed values as a pandas.DataFrame object. If ``write_output`` is True, then it saves the imputed data as a CSV file called ``<prefix>_imputed_012.csv``.

        output_format (str, optional): Format of output imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list. Defaults to "df".

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        **kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool}. "iterative_mode" determines whether ``ImputeAlleleFreq`` is being used as the initial imputer in ``IterativeImputer``.

    Raises:
        TypeError: genotype_data and gt cannot both be NoneType.
        TypeError: genotype_data and gt cannot both be provided.
        TypeError: Either pops or genotype_data must be defined if by_populations is True.
    """

    def __init__(
        self,
        *,
        genotype_data: Any,
        latent_features: int = 2,
        n_steps: int = 5000,
        learning_rate: float = 0.0002,
        regularization_param: float = 0.02,
        missing: int = -9,
        prefix: str = "output",
        write_output: bool = True,
        output_format: str = "df",
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:

        super().__init__()

        if genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")

        self.n_steps = n_steps
        self.latent_features = latent_features
        self.regularization_param = regularization_param
        self.learning_rate = learning_rate
        self.missing = missing
        self.prefix = prefix
        self.output_format = output_format
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)
        print(genotype_data)
        self.imputed = self.fit_predict(genotype_data)

        # if write_output:
        #     self.write2file(self.imputed)

    def fit_predict(self, X):
        # print(X)
        # if isinstance(X, (list, np.ndarray)):
        #     df = pd.DataFrame(X)
        # elif isinstance(X, pd.DataFrame):
        #     df = X.copy()
        # else:
        #     raise TypeError(
        #         f"X must be of type list(list(int)), numpy.ndarray, "
        #         f"or pandas.DataFrame, but got {type(X)}"
        #     )

        #df.replace(self.missing, np.nan, inplace=True)

        #data = pd.DataFrame()

        #imputation
        #print(data)
        R = X
        R = R+1
        R[R<0] = 0
        print(R)
        n_row = len(R)
        n_col = len(R[0])
        p = np.random.rand(n_row,self.latent_features)
        q = np.random.rand(n_col,self.latent_features)
        q_t = q.T

        for step in range(self.n_steps):
            for i in range(n_row):
                for j in range(n_col):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(p[i,:],q_t[:,j])
                        for k in range(self.latent_features):
                            p[i][k] = p[i][k] + self.learning_rate * (2 * eij * q_t[k][j] - self.regularization_param * p[i][k])
                            q_t[k][j] = q_t[k][j] + self.learning_rate * (2 * eij * p[i][k] - self.regularization_param * q_t[k][j])
            eR = np.dot(p,q_t)
            e = 0
            for i in range(n_row):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(p[i,:],q_t[:,j]), 2)
                        for k in range(self.latent_features):
                            e = e + (self.regularization_param/2) * ( pow(p[i][k],2) + pow(q_t[k][j],2) )
            if e < 0.001:
                break
        nR = np.dot(p, q_t)
        print(nR)

        #transform values per-column (i.e., only allowing values found in original)
        tR=nR
        for j in range(n_col):
            observed = nR[:,j]
            expected = R[:,j]
            options = np.unique(expected[expected != 0])
            for i in range(n_row):
                transform = min(options, key=lambda x:abs(x-nR[i,j]))
                tR[i,j]=transform
        tR = tR-1
        tR[tR<0] = -9
        print(tR)

        #get accuracy of re-constructing non-missing genotypes
        diff=np.sum(X[X>=0]==tR[X>=0])
        tot=X[X>=0].size
        accuracy=diff/tot
        print(accuracy)

        #insert imputed values for missing genotypes
        fR=X
        fR[X<0] = tR[X<0]
        print(fR)

        # if self.iterative_mode:
        #     data = data.astype(dtype="float32")
        # else:
        #     data = data.astype(dtype="Int8")
        #
        # if self.verbose:
        #     print("Done!")
        #
        # if self.output_format == "df":
        #     return data
        #
        # elif self.output_format == "array":
        #     return data.to_numpy()
        #
        # elif self.output_format == "list":
        #     return data.values.tolist()

        # else:
        #     raise ValueError("Unknown output_format type specified!")

    # def write2file(
    #     self, X: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]
    # ) -> None:
    #     """Write imputed data to file on disk.
    #
    #     Args:
    #         X (pandas.DataFrame, numpy.ndarray, List[List[Union[int, float]]]): Imputed data to write to file.
    #
    #     Raises:
    #         TypeError: If X is of unsupported type.
    #     """
    #     outfile = f"{self.prefix}_imputed_012.csv"
    #
    #     if isinstance(X, pd.DataFrame):
    #         df = X
    #     elif isinstance(X, (np.ndarray, list)):
    #         df = pd.DataFrame(X)
    #     else:
    #         raise TypeError(
    #             f"Could not write imputed data because it is of incorrect "
    #             f"type. Got {type(X)}"
    #         )
    #
    #     df.to_csv(outfile, header=False, index=False)
