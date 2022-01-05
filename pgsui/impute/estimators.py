# Standard library imports
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
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
try:
    from ..read_input.read_input import GenotypeData

    from .impute import Impute
    from .neural_network_imputers import VAE, UBP

    from ..utils import misc
    from ..utils.misc import get_processor_name
    from ..utils.misc import isnotebook
    from ..utils.misc import timer
except (ModuleNotFoundError, ValueError):
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


class ImputeKNN(Impute):
    """Does K-Nearest Neighbors Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

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

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

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

        str_encodings (dict(str: int), optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "n_neighbors": Integer(3, 10),
        >>>    "leaf_size": Integer(10, 50),
        >>>}
        >>>
        >>>knn = ImputeKNN(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     ga=True,
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>knn_gtdata = knn.imputed
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

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeRandomForest(Impute):
    """Does Random Forest or Extra Trees Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        write_output (bool, optional): If True, writes imputed data to file on disk. Otherwise just stores it as a class attribute.

        gridparams (Dict[str, Any] or None or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV), the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

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

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        extra_trees (bool, optional): Whether to use ExtraTreesClassifier (If True) instead of RandomForestClassifier (If False). ExtraTreesClassifier is faster, but is not supported by the scikit-learn-intelex patch, whereas RandomForestClassifier is. If using an Intel CPU, the optimizations provided by the scikit-learn-intelex patch might make setting ``extratrees=False`` worthwhile. If you are not using an Intel CPU, the scikit-learn-intelex library is not supported and ExtraTreesClassifier will be faster with similar performance. NOTE: If using scikit-learn-intelex, ``criterion`` must be set to "gini" and ``oob_score`` to False, as those parameters are not currently supported herein. Defaults to True.

        n_estimators (int, optional): The number of trees in the forest. Increasing this value can improve the fit, but at the cost of compute time and resources. Defaults to 100.

        criterion (str, optional): The function to measure the quality of a split. Supported values are "gini" for the Gini impurity and "entropy" for the information gain. Defaults to "gini".

        max_depth (int, optional): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. Defaults to None.

        min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. If value is an integer, then considers min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples), rounded up to the nearest integer, are the minimum number of samples for each split. Defaults to 2.

        min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, then ``min_samples_leaf`` is the minimum number. If value is floating point, then ``min_samples_leaf`` is a fraction and ``int(min_samples_leaf * n_samples)`` is the minimum number of samples for each node. Defaults to 1.

        min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided. Defaults to 0.0.

        max_features (str, int, float, or None, optional): The number of features to consider when looking for the best split. If int, then consider "max_features" features at each split. If float, then "max_features" is a fraction and ``int(max_features * n_samples)`` features are considered at each split. If "auto", then ``max_features=sqrt(n_features)``\. If "sqrt", then ``max_features=sqrt(n_features)``\. If "log2", then ``max_features=log2(n_features)``\. If None, then ``max_features=n_features``\. Defaults to "auto".

        max_leaf_nodes (int or None, optional): Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. Defaults to None.

        min_impurity_decrease (float, optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. See ``sklearn.ensemble.ExtraTreesClassifier`` documentation for more information. Defaults to 0.0.

        bootstrap (bool, optional): Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree. Defaults to False.

        oob_score (bool, optional): Whether to use out-of-bag samples to estimate the generalization score. Only available if ``bootstrap=True``\. Defaults to False.

        max_samples (int or float, optional): If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draws ``X.shape[0] samples``\. if int, then draws ``max_samples`` samples. If float, then draws ``int(max_samples * X.shape[0] samples)`` with ``max_samples`` in the interval (0, 1). Defaults to None.

        clf_random_state (int or None, optional): Controls three sources of randomness for ``sklearn.ensemble.ExtraTreesClassifier``: The bootstrapping of the samples used when building trees (if ``bootstrap=True``), the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``), and the draw of the splits for each of the ``max_features``\. If None, then uses a different random seed each time the imputation is run. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population based on a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "min_samples_leaf": Integer(1, 10),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>rf = ImputeRandomForest(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     ga=True,
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>rf_gtdata = rf.imputed
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

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeGradientBoosting(Impute):
    """Does Gradient Boosting Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV), the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

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

        early_stop_gen (int, optional): If the genetic algorithm sees ``early_stop_gen`` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform. Defaults to 5.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        column_subset (int or float, optional): If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ``column_subset`` columns. If float, subset ``int(n_features * column_subset)`` columns. Defaults to 0.1.

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors. Defaults to 1.

        n_estimators (int, optional): The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance but also increases compute time and required resources. Defaults to 100.

        loss (str, optional): The loss function to be optimized. "deviance" refers to deviance (=logistic regression) for classification with probabilistic outputs. For loss "exponential" gradient boosting recovers the AdaBoost algorithm. Defaults to "deviance".

        learning_rate (float, optional): Learning rate shrinks the contribution of each tree by ``learning_rate``\. There is a trade-off between ``learning_rate`` and ``n_estimators``\. Defaults to 0.1.

        subsample (float, optional): The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. ``subsample`` interacts with the parameter ``n_estimators``\. Choosing ``subsample < 1.0`` leads to a reduction of variance and an increase in bias. Defaults to 1.0.

        criterion (str, optional): The function to measure the quality of a split. Supported criteria are "friedman_mse" for the mean squared error with improvement score by Friedman and "mse" for mean squared error. The default value of "friedman_mse" is generally the best as it can provide a better approximation in some cases. Defaults to "friedman_mse".

        min_samples_split (int or float, optional): The minimum number of samples required to split an internal node. If value is an integer, then consider ``min_samples_split`` as the minimum number. If value is a floating point, then min_samples_split is a fraction and ``(min_samples_split * n_samples)`` is rounded up to the nearest integer and used as the number of samples per split. Defaults to 2.

        min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, consider ``min_samples_leaf`` as the minimum number. If value is a floating point, then ``min_samples_leaf`` is a fraction and ``(min_samples_leaf * n_samples)`` rounded up to the nearest integer is the minimum number of samples per node. Defaults to 1.

        min_weight_fraction_leaf (float, optional): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when ``sample_weight`` is not provided. Defaults to 0.0.

        max_depth (int, optional): The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables. Defaults to 3.

        min_impurity_decrease (float, optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Defaults to 0.0. See ``sklearn.ensemble.GradientBoostingClassifier`` documentation for more information. Defaults to 0.0.

        max_features (int, float, str, or None, optional): The number of features to consider when looking for the best split. If value is an integer, then consider ``max_features`` features at each split. If value is a floating point, then ``max_features`` is a fraction and ``(max_features * n_features)`` is rounded to the nearest integer and considered as the number of features per split. If "auto", then ``max_features=sqrt(n_features)``\. If "sqrt", then ``max_features=sqrt(n_features)``\. If "log2", then ``max_features=log2(n_features)``\. If None, then ``max_features=n_features``\. Defaults to None.

        max_leaf_nodes (int or None, optional): Grow trees with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then uses an unlimited number of leaf nodes. Defaults to None.

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split. Pass an int for reproducible output across multiple function calls. If None, then uses a different random seed for each function call. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "learning_rate": Continuous(lower=0.01, upper=0.1),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>gb = ImputeGradientBoosting(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     ga=True,
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>gb_gtdata = gb.imputed
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

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeBayesianRidge(Impute):
    """NOTE: This is a regressor estimator and is only intended for testing purposes, as it is faster than the classifiers. Does Bayesian Ridge Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

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

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

        n_jobs (int, optional): Number of parallel jobs to use. If ``gridparams`` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors. Defaults to 1.

        n_iter (int, optional): Maximum number of iterations. Should be greater than or equal to 1. Defaults to 300.

        clf_tol (float, optional): Stop the algorithm if w has converged. Defaults to 1e-3.

        alpha_1 (float, optional): Hyper-parameter: shape parameter for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.

        alpha_2 (float, optional): Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.

        lambda_1 (float, optional): Hyper-parameter: shape parameter for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.

        lambda_2 (float, optional): Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.

        alpha_init (float or None, optional): Initial value for alpha (precision of the noise). If None, ``alpha_init`` is ``1/Var(y)``\. Defaults to None.

        lambda_init (float or None, optional): Initial value for lambda (precision of the weights). If None, ``lambda_init`` is 1. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "alpha_1": Continuous(
        >>>         lower=1e-6, upper=0.1, distribution="log-uniform"
        >>>     ),
        >>>     "lambda_1": Continuous(
        >>>         lower=1e-6, upper=0.1, distribution="log-uniform"
        >>>     ),
        >>>}
        >>>
        >>>br = ImputeBayesianRidge(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     ga=True,
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>br_gtdata = br.imputed
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

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeXGBoost(Impute):
    """Does XGBoost (Extreme Gradient Boosting) Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process. The grid searches are not compatible with XGBoost, but validation scores can still be calculated without a grid search. In addition, ImputeLightGBM is a similar algorithm and is compatible with grid searches, so use ImputeLightGBM if you want a grid search.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        cv (int, optional): Number of folds for cross-validation during randomized grid search. Defaults to 5.

        validation_only (float or None, optional): Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``int(n_features * validation_only)`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None. Defaults to 0.4.

        scoring_metric (str, optional): Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options. Defaults to "accuracy".

        chunk_size (int or float, optional): Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ``chunk_size`` loci at a time. If a float is specified, selects ``math.ceil(total_loci * chunk_size)`` loci at a time. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used. Defaults to False.

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features. Defaults to None.

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

        clf_random_state (int, numpy.random.RandomState object, or None, optional): Controls three sources of randomness for ``sklearn.ensemble.ExtraTreesClassifier``: The bootstrapping of the samples used when building trees (if ``bootstrap=True``), the sampling of the features to consider when looking for the best split at each node (if ``max_features < n_features``), and the draw of the splits for each of the ``max_features``\. If None, then uses a different random seed each time the imputation is run. Defaults to None.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int or None, optional): Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if ``n_nearest_features`` is not None or the imputation_order is "random". Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "learning_rate": Continuous(lower=0.01, upper=0.1),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>xgb = ImputeXGBoost(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     ga=True,
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>xgb_gtdata = xgb.imputed
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

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeLightGBM(Impute):
    """Does LightGBM (Light Gradient Boosting) Iterative imputation of missing data. LightGBM is an alternative to XGBoost that is around 7X faster and uses less memory, while still maintaining high accuracy. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance that was used to read in the sequence data.

        prefix (str, optional): Prefix for imputed data's output filename.

        gridparams (Dict[str, Any] or None, optional): Dictionary with keys=keyword arguments for the specified estimator and values=lists of parameter values or distributions. If using RandomizedSearchCV, distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ``gridparams`` will be used for a randomized grid search with cross-validation. If using the genetic algorithm grid search (GASearchCV) by setting ``ga=True``\, the parameters can be specified as ``sklearn_genetic.space`` objects. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. If ``gridparams=None``\, a grid search is not performed. Defaults to None.

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

        progress_update_percent (int or None, optional): Print status updates for features every ``progress_update_percent``\%. IterativeImputer iterations will always be printed, but ``progress_update_percent`` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

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

        str_encodings (Dict[str, int], optional): Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``\. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (from features with fewest missing values to most), "descending" (from features with most missing values to fewest), "roman" (left to right), "arabic" (right to left), "random" (a random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time. Defaults to False.

        random_state (int or None, optional): The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time. Defaults to None.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>from sklearn_genetic.space import Categorical, Integer, Continuous
        >>>
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>># Genetic Algorithm grid_params
        >>>grid_params = {
        >>>    "learning_rate": Continuous(lower=0.01, upper=0.1),
        >>>    "max_depth": Integer(2, 110),
        >>>}
        >>>
        >>>lgbm = ImputeLightGBM(
        >>>     genotype_data=data,
        >>>     gridparams=grid_params,
        >>>     cv=5,
        >>>     ga=True,
        >>>     n_nearest_features=10,
        >>>     n_estimators=100,
        >>>     initial_strategy="phylogeny",
        >>>)
        >>>
        >>>lgbm_gtdata = lgbm.imputed
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

        super().__init__(self.clf, self.clf_type, kwargs)

        self.imputed, self.best_params = self.fit_predict(
            genotype_data.genotypes012_df
        )


class ImputeVAE(Impute):
    """Class to impute missing data using a Variational Autoencoder neural network model.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

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

        dropout_probability (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

        hidden_activation (str): Activation function to use for hidden layers. See tf.keras.activations for more info. Defaults to "relu".

        output_activation (str): Activation function to use for output layer. See tf.keras.activations for more info. Defaults to "sigmoid".

        kernel_initializer (str): Initializer to use for initializing model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float): L1 regularization penalty to apply. Adjust if overfitting is occurring. Defaults to 0.

        l2_penalty (float): L2 regularization penalty to apply. Adjust if overfitting is occurring. Defaults to 0.

    Attributes:
        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>>vae = ImputeVAE(
        >>>     genotype_data=data,
        >>>     cv=5,
        >>>     learning_rate=0.05,
        >>>     validation_only=0.3
        >>>)
        >>>
        >>>vae_gtdata = vae.imputed
    """

    def __init__(
        self,
        genotype_data,
        *,
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

        super().__init__(self.clf, self.clf_type, all_kwargs)

        if genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")

        X = genotype_data.genotypes012_array

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self.imputed, self.best_params = self.fit_predict(df)


class ImputeUBP(Impute):
    """Class to impute missing data using unsupervised backpropagation neural network models.

    UBP [1]_ is an extension of NLPCA [2]_ with the input being randomly generated and of reduced dimensionality that gets trained to predict the supplied output based on only known values. It then uses the trained model to predict missing values. However, in contrast to NLPCA, UBP trains the model over three phases. The first is a single layer perceptron used to refine the randomly generated input. The second phase is a multi-layer perceptron that uses the refined reduced-dimension data from the first phase as input. In the second phase, the model weights are refined but not the input. In the third phase, the model weights and the inputs are then refined.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

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

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``\. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", and "adagrad" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        max_epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        tol (float, optional): Tolerance level to use for the early stopping criterion. If the loss does not improve past the tolerance level after ``early_stop_gen`` epochs, then training will halt. Defaults to 1e-3.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional) L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        dropout_probability (float, optional): Dropout rate during training to reduce overfitting. Must be a float between 0 and 1. Defaults to 0.2.

    Attributes:
        nlpca (bool): If True, does NLPCA model. Otherwise does UBP.

        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>> ubp = ImputeUBP(
        >>>     genotype_data=data,
        >>>     cv=5,
        >>>     learning_rate=0.05,
        >>>     validation_only=0.3
        >>>)
        >>>
        >>>ubp_gtdata = ubp.imputed

    References:
        .. [1] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.

        .. [2] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.


    """

    nlpca = False

    def __init__(
        self,
        genotype_data,
        *,
        prefix="output",
        gridparams=None,
        grid_iter=50,
        cv=5,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=25,
        scoring_metric="neg_mean_squared_error",
        column_subset=0.1,
        initial_strategy="populations",
        validation_only=0.3,
        write_output=True,
        disable_progressbar=False,
        chunk_size=1.0,
        n_jobs=1,
        batch_size=32,
        n_components=3,
        num_hidden_layers=3,
        hidden_layer_sizes="midpoint",
        optimizer="adam",
        hidden_activation="elu",
        learning_rate=0.1,
        max_epochs=100,
        tol=1e-3,
        weights_initializer="glorot_normal",
        l1_penalty=0.01,
        l2_penalty=0.01,
        dropout_probability=0.2,
    ):

        # Get local variables into dictionary object
        settings = locals()
        settings["nlpca"] = self.nlpca

        self.clf = UBP
        self.clf_type = "classifier"
        if self.nlpca:
            self.clf.__name__ = "NLPCA"

        imp_kwargs = {
            "str_encodings": {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        }

        settings.update(imp_kwargs)

        if genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")

        X = genotype_data.genotypes012_array

        super().__init__(self.clf, self.clf_type, settings)

        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self.imputed, self.best_params = self.fit_predict(df)


class ImputeNLPCA(ImputeUBP):
    """Class to impute missing data using inverse non-linear principal component analysis (NLPCA) neural network models.

    NLPCA [1]_ trains randomly generated, reduced-dimensionality input to predict the correct output. In the case of imputation, the model is trained only on known values, and the trained model is then used to predict the missing values.

    Args:
        genotype_data (GenotypeData object): Input data initialized as GenotypeData object. Required positional argument.

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

        hidden_layer_sizes (str, List[int], List[str], or int, optional): Number of neurons to use in hidden layers. If string or a list of strings is supplied, the strings must be either "midpoint", "sqrt", or "log2". "midpoint" will calculate the midpoint as ``(n_features + n_components) / 2``\. If "sqrt" is supplied, the square root of the number of features will be used to calculate the output units. If "log2" is supplied, the units will be calculated as ``log2(n_features)``\. hidden_layer_sizes will calculate and set the number of output units for each hidden layer. If one string or integer is supplied, the model will use the same number of output units for each hidden layer. If a list of integers or strings is supplied, the model will use the values supplied in the list, which can differ. The list length must be equal to the ``num_hidden_layers``\. Defaults to "midpoint".

        optimizer (str, optional): The optimizer to use with gradient descent. Possible value include: "adam", "sgd", and "adagrad" are supported. See tf.keras.optimizers for more info. Defaults to "adam".

        hidden_activation (str, optional): The activation function to use for the hidden layers. See tf.keras.activations for more info. Commonly used activation functions include "elu", "relu", and "sigmoid". Defaults to "elu".

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        max_epochs (int, optional): Maximum number of epochs to run if the ``early_stop_gen`` criterion is not met.

        tol (float, optional): Tolerance level to use for the early stopping criterion. If the loss does not improve past the tolerance level after ``early_stop_gen`` epochs, then training will halt. Defaults to 1e-3.

        weights_initializer (str, optional): Initializer to use for the model weights. See tf.keras.initializers for more info. Defaults to "glorot_normal".

        l1_penalty (float, optional): L1 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

        l2_penalty (float, optional) L2 regularization penalty to apply to reduce overfitting. Defaults to 0.01.

    Attributes:
        nlpca (bool): If True, does NLPCA model. Otherwise does UBP.

        clf (sklearn or neural network classifier): Estimator to use.

        imputed (GenotypeData): New GenotypeData instance with imputed data.

        best_params (Dict[str, Any]): Best found parameters from grid search. In neural networks this value is None because grid searches are not supported.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>> nlpca = ImputeNLPCA(
        >>>     genotype_data=data,
        >>>     cv=5,
        >>>     learning_rate=0.05,
        >>>     validation_only=0.3
        >>>)
        >>>
        >>>nlpca_gtdata = nlpca.imputed

    References:
    .. [1] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.
    """

    nlpca = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
