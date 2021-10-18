# Standard library imports
import sys
from timeit import default_timer

# Third-party imports
import numpy as np
import pandas as pd
import scipy.linalg
import toyplot.pdf
import toyplot as tp
import toytree as tt

# Scikit-learn imports
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
import lightgbm as lgbm

import theano
import theano.tensor as T
import theano.tensor.extra_ops
import theano.tensor.nnet as nnet

# Custom imports
from read_input.read_input import GenotypeData
from impute.impute import Impute

from utils import misc
from utils.misc import get_processor_name
from utils.misc import gradient_descent
from utils.misc import initialize_weights
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
    """[Does K-Nearest Neighbors Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_neighbors (int, optional): [Number of neighbors to use by default for K-Nearest Neighbors queries]. Defaults to 5.

        weights (str, optional): [Weight function used in prediction. Possible values: 'Uniform': Uniform weights with all points in each neighborhood weighted equally; 'distance': Weight points by the inverse of their distance, in this case closer neighbors of a query point will have  a greater influence than neighbors that are further away; 'callable': A user-defined function that accepts an array of distances and returns an array of the same shape containing the weights]. Defaults to "distance".

        algorithm (str, optional): [Algorithm used to compute the nearest neighbors. Possible values: 'ball_tree', 'kd_tree', 'brute', 'auto']. Defaults to "auto".

        leaf_size (int, optional): [Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem]. Defaults to 30.

        p (int, optional): [Power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance (l1), and if p=2 it is equivalent to using euclidean distance (l2). For arbitrary p, minkowski_distance (l_p) is used]. Defaults to 2.

        metric (str, optional): [The distance metric to use for the tree. The default metric is minkowski, and with p=2 this is equivalent to the standard Euclidean metric. See the documentation of sklearn.DistanceMetric for a list of available metrics. If metric is 'precomputed', X is assumed to be a distance matrix and must be square during fit]. Defaults to "minkowski".

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): [Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``]. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.

    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_neighbors=5,
        weights="distance",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        str_encodings={"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
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
    """[Does Random Forest Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        extra_trees (bool, optional): [Whether to use ExtraTreesClassifier (If True) instead of RandomForestClassifier (If False). ExtraTreesClassifier is faster, but is not supported by the scikit-learn-intelex patch and RandomForestClassifier is. If using an Intel CPU, the optimizations provided by the scikit-learn-intelex patch might make setting ```extratrees=False``` worthwhile. If you are not using an Intel CPU, the scikit-learn-intelex library is not supported and ExtraTreesClassifier will be faster with similar performance. NOTE: If using scikit-learn-intelex, ```criterion``` must be set to "gini" and ```oob_score``` to False, as those parameters are not currently supported]. Defaults to True.

        n_estimators (int, optional): [The number of trees in the forest. Increasing this value can improves the fit, but at the cost of compute time]. Defaults to 100.

        criterion (str, optional): [The function to measure the quality of a split. Supported values are 'gini' for the Gini impurity and 'entropy' for the information gain]. Defaults to "gini".

        max_depth (int, optional): [The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples]. Defaults to None.

        min_samples_split (int or float, optional): [The minimum number of samples required to split an internal node. If value is an integer, then considers min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples), rounded up to the nearest integer, are the minimum number of samples for each split]. Defaults to 2.

        min_samples_leaf (int or float, optional): [The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, then min_samples_leaf is the minimum number. If value is floating point, then min_samples_leaf is a fraction and (min_samples_leaf * n_samples) rounded up to the nearest integer is the minimum number of samples for each node]. Defaults to 1.

        min_weight_fraction_leaf (float, optional): [The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.]. Defaults to 0.0.

        max_features (int or float, optional): [The number of features to consider when looking for the best split. If int, then consider 'max_features' features at each split. If float, then 'max_features' is a fraction and (max_features * n_samples) features, rounded to the nearest integer, are considered at each split. If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features]. Defaults to "auto".

        max_leaf_nodes (int, optional): [Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes]. Defaults to None.

        min_impurity_decrease (float, optional): [A node will be split if this split induces a decrease of the impurity greater than or equal to this value. See sklearn.ensemble.ExtraTreesClassifier documentation for more information]. Defaults to 0.0.

        bootstrap (bool, optional): [Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree]. Defaults to False.

        oob_score (bool, optional): [Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True]. Defaults to False.

        max_samples (int or float, optional): [If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draws X.shape[0] samples. if int, then draw 'max_samples' samples. If float, then draw (max_samples * X.shape[0] samples) with max_samples in the interval (0, 1)]. Defaults to None.

        clf_random_state (int, optional): [Controls three sources of randomness for sklearn.ensemble.ExtraTreesClassifier: The bootstrapping of the samples used when building trees (if bootstrap=True), the sampling of the features to consider when looking for the best split at each node (if max_features < n_features), and the draw of the splits for each of the max_features. If None, then uses a different random seed each time the imputation is run]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): [Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``]. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        extratrees=True,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        max_samples=None,
        clf_random_state=None,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        str_encodings={"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
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
    """[Does Gradient Boosting Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_estimators (int, optional): [The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance]. Defaults to 100.

        loss (str, optional): [The loss function to be optimized. ‘deviance’ refers to deviance (=logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm]. Defaults to "deviance".

        learning_rate (float, optional): [Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators]. Defaults to 0.1.

        subsample (float, optional): [The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias]. Defaults to 1.0.

        criterion (str, optional): [The function to measure the quality of a split. Supported criteria are 'friedman_mse' for the mean squared error with improvement score by Friedman and 'mse' for mean squared error. The default value of 'friedman_mse' is generally the best as it can provide a better approximation in some cases]. Defaults to "friedman_mse".

        min_samples_split (int or float, optional): [The minimum number of samples required to split an internal node. If value is an integer, then consider min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples) is rounded up to the nearest integer and used as the number of samples per split]. Defaults to 2.

        min_samples_leaf (int or float, optional): [The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, consider min_samples_leaf as the minimum number. If value is a floating point, then min_samples_leaf is a fraction and (min_samples_leaf * n_samples) rounded up to the nearest integer is the minimum number of samples per node]. Defaults to 1.

        min_weight_fraction_leaf (float, optional): [The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided]. Defaults to 0.0.

        max_depth (int, optional): [The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.]. Defaults to 3.

        min_impurity_decrease (float, optional): [A node will be split if this split induces a decrease of the impurity greater than or equal to this value]. Defaults to 0.0. See sklearn.ensemble.GradientBoostingClassifier documentation for more information]. Defaults to 0.0.

        max_features (int, float, or str, optional): [The number of features to consider when looking for the best split. If value is an integer, then consider 'max_features' features at each split. If value is a floating point, then 'max_features' is a fraction and (max_features * n_features) is rounded to the nearest integer and considered as the number of features per split. If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features]. Defaults to None.

        max_leaf_nodes (int, optional): [Grow trees with 'max_leaf_nodes' in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then uses an unlimited number of leaf nodes]. Defaults to None.

        clf_random_state (int, optional): [Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split. Pass an int for reproducible output across multiple function calls. If None, then uses a different random seed for each function call]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): [Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``]. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_estimators=100,
        loss="deviance",
        learning_rate=0.1,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        max_features=None,
        max_leaf_nodes=None,
        clf_random_state=None,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        str_encodings={"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
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
    """[NOTE: This is a regressor estimator and is only intended for testing purposes, as it is faster than the classifiers. Does Bayesian Ridge Iterative Imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search:     If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the regressor. -1 means using all available processors]. Defaults to 1.

        n_iter (int, optional): [Maximum number of iterations. Should be greater than or equal to 1]. Defaults to 300.

        clf_tol (float, optional): [Stop the algorithm if w has converged]. Defaults to 1e-3.

        alpha_1 (float, optional): [Hyper-parameter: shape parameter for the Gamma distribution prior over the alpha parameter]. Defaults to 1e-6.

        alpha_2 (float, optional): [Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter]. Defaults to 1e-6.

        lambda_1 (float, optional): [Hyper-parameter: shape parameter for the Gamma distribution prior over the lambda parameter]. Defaults to 1e-6.

        lambda_2 (float, optional): [Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter]. Defaults to 1e-6.

        alpha_init (float, optional): [Initial value for alpha (precision of the noise). If None, alpha_init is 1/Var(y).]. Defaults to None.

        lambda_init (float, optional): [Initial value for lambda (precision of the weights). If None, lambda_init is 1]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): [Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``]. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="neg_root_mean_squared_error",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_iter=300,
        clf_tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        alpha_init=None,
        lambda_init=None,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        str_encodings={"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
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
    """[Does XGBoost (Extreme Gradient Boosting) Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process. The grid searches are not compatible with XGBoost, but validation scores can still be calculated without a grid search. In addition, ImputeLightGBM is a similar algorithm and is compatible with grid searches, so use ImputeLightGBM if you want a grid search]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_estimators (int, optional): [The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage]. Defaults to 100.

        max_depth (int, optional): [Maximum tree depth for base learners]. Defaults to 3.

        learning_rate (float, optional): [Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources]. Defaults to 0.1.

        booster (str, optional): [Specify which booster to use. Possible values include "gbtree", "gblinear", and "dart"]. Defaults to "gbtree".

        gamma (float, optional): [Minimum loss reduction required to make a further partition on a leaf node of the tree]. Defaults to 0.0.

        min_child_weight (float, optional): [Minimum sum of instance weight(hessian) needed in a child]. Defaults to 1.0.

        max_delta_step (float, optional): [Maximum delta step we allow each tree's weight estimation to be]. Defaults to 0.0.

        subsample (float, optional): [Subsample ratio of the training instance]. Defaults to 1.0.

        colsample_bytree (float, optional): [Subsample ratio of columns when constructing each tree]. Defaults to 1.0.

        reg_lambda (float, optional): [L2 regularization term on weights (xgb's lambda parameter)]. Defaults to 1.0.

        reg_alpha (float, optional): [L1 regularization term on weights (xgb's alpha parameter)]. Defaults to 1.0.

        clf_random_state (int or numpy.random.RandomState object, optional): [Controls three sources of randomness for sklearn.ensemble.ExtraTreesClassifier: The bootstrapping of the samples used when building trees (if bootstrap=True), the sampling of the features to consider when looking for the best split at each node (if max_features < n_features), and the draw of the splits for each of the max_features. If None, then uses a different random seed each time the imputation is run]. Defaults to None.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): [Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``]. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        cv=5,
        validation_only=0.4,
        scoring_metric="accuracy",
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        booster="gbtree",
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        reg_lambda=1,
        reg_alpha=0,
        clf_random_state=None,
        n_nearest_features=10,
        max_iter=10,
        tol=1e-3,
        initial_strategy="most_frequent",
        str_encodings={"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):
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
    """[Does LightGBM (Light Gradient Boosting) Iterative imputation of missing data. LightGBM is an alternative to XGBoost that is around 7X faster and uses less memory, while still maintaining high accuracy. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

        prefix ([str]): [Prefix for imputed data's output filename]

        gridparams (dict, optional): [Dictionary with lists as values or distributions of parameters. Distributions can be specified by using scipy.stats.uniform(low, high) (for a uniform distribution) or scipy.stats.loguniform(low, high) (useful if range of values spans orders of magnitude). ```gridparams``` will be used for a randomized grid search. The grid search will determine the optimal parameters as those that maximize accuracy (or minimize root mean squared error for BayesianRidge regressor). **NOTE: Takes a long time, so run it with a small subset of the data just to find the optimal parameters for the classifier, then run a full imputation using the optimal parameters. The full imputation can be performed by setting ```gridparams=None``` (default)]. Defaults to None.

        grid_iter (int, optional): [Number of iterations for randomized grid search]. Defaults to 50.

        cv (int, optional): [Number of folds for cross-validation during randomized grid search]. Defaults to 5.

        validation_only (float, optional): [Validates the imputation without doing a grid search. The validation method randomly replaces 15% to 50% of the known, non-missing genotypes in ``n_features * validation_only`` of the features. It then imputes the newly missing genotypes for which we know the true values and calculates validation scores. This procedure is replicated ``cv`` times and a mean, median, minimum, maximum, lower 95% confidence interval (CI) of the mean, and the upper 95% CI are calculated and saved to a CSV file. ``gridparams`` must be set to None (default) for ``validation_only`` to work. Calculating a validation score can be turned off altogether by setting ``validation_only`` to None]. Defaults to 0.4.

        ga (bool, optional): [Whether to use a genetic algorithm for the grid search. If False, a RandomizedSearchCV is done instead]. Defaults to False.

        population_size (int, optional): [For genetic algorithm grid search: Size of the initial population to sample randomly generated individuals. See GASearchCV documentation]. Defaults to 10.

        tournament_size (int, optional): [For genetic algorithm grid search: Number of individuals to perform tournament selection. See GASearchCV documentation]. Defaults to 3.

        elitism (bool, optional): [For genetic algorithm grid search: If True takes the tournament_size best solution to the next generation. See GASearchCV documentation]. Defaults to True.

        crossover_probability (float, optional): [For genetic algorithm grid search: Probability of crossover operation between two individuals. See GASearchCV documentation]. Defaults to 0.8.

        mutation_probability (float, optional): [For genetic algorithm grid search: Probability of child mutation. See GASearchCV documentation]. Defaults to 0.1.

        ga_algorithm (str, optional): [For genetic algorithm grid search: Evolutionary algorithm to use. See more details in the deap algorithms documentation]. Defaults to "eaMuPlusLambda".

        early_stop_gen (int, optional): [If the genetic algorithm sees ```early_stop_gen``` consecutive generations without improvement in the scoring metric, an early stopping callback is implemented. This saves time by reducing the number of generations the genetic algorithm has to perform]. Defaults to 5.

        scoring_metric (str, optional): [Scoring metric to use for randomized or genetic algorithm grid searches. See https://scikit-learn.org/stable/modules/model_evaluation.html for supported options]. Defaults to "accuracy".

        column_subset (int or float, optional): [If float, proportion of the dataset to randomly subset for the grid search. Should be between 0 and 1, and should also be small, because the grid search takes a long time. If int, subset ```column_subset``` columns]. Defaults to 0.1.

        chunk_size (int or float, optional): [Number of loci for which to perform IterativeImputer at one time. Useful for reducing the memory usage if you are running out of RAM. If integer is specified, selects ```chunk_size``` loci at a time. If a float is specified, selects ```total_loci * chunk_size``` loci at a time]. Defaults to 1.0 (all features).

        disable_progressbar (bool, optional): [Whether or not to disable the tqdm progress bar when doing the imputation. If True, progress bar is disabled, which is useful when running the imputation on e.g. an HPC cluster. If the bar is disabled, a status update will be printed to standard output for each iteration and feature instead. If False, the tqdm progress bar will be used]. Defaults to False.

        progress_update_percent (int, optional): [Print status updates for features every ```progress_update_percent```%. IterativeImputer iterations will always be printed, but ```progress_update_percent``` involves iteration progress through the features of each IterativeImputer iteration. If None, then does not print progress through features]. Defaults to None.

        n_jobs (int, optional): [Number of parallel jobs to use. If ```gridparams``` is not None, n_jobs is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors]. Defaults to 1.

        n_estimators (int, optional): [The number of boosting rounds. Increasing this value can improve the fit, but at the cost of compute time and RAM usage]. Defaults to 100.

        max_depth (int, optional): [Maximum tree depth for base learners]. Defaults to 3.

        learning_rate (float, optional): [Boosting learning rate (eta). Basically, it serves as a weighting factor for correcting new trees when they are added to the model. Typical values are between 0.1 and 0.3. Lower learning rates generally find the best optimum at the cost of requiring far more compute time and resources]. Defaults to 0.1.

        boosting_type (str, optional): [Possible values: "gbdt", traditional Gradient Boosting Decision Tree. "dart", Dropouts meet Multiple Additive Regression Trees. "goss", Gradient-based One-Side Sampling. The "rf" option is not currently supported]. Defaults to "gbdt".

        num_leaves (int, optional): [Maximum tree leaves for base learners]. Defaults to 31.

        subsample_for_bin (int, optional): [Number of samples for constructing bins]. Defaults to 200000.

        min_split_gain (float, optional): [Minimum loss reduction required to make a further partition on a leaf node of the tree]. Defaults to 0.0.

        min_child_weight (float, optional): [Minimum sum of instance weight (hessian) needed in a child (leaf)]. Defaults to 1e-3.

        min_child_samples (int, optional): [Minimum number of data needed in a child (leaf)]. Defaults to 20.

        subsample (float, optional): [Subsample ratio of the training instance]. Defaults to 1.0.

        subsample_freq (int, optional): [Frequency of subsample, <=0 means no enable]. Defaults to 0.

        colsample_bytree (float, optional): [Subsample ratio of columns when constructing each tree]. Defaults to 1.0.

        reg_lambda (float, optional): [L2 regularization term on weights]. Defaults to 0.

        reg_alpha (float, optional): [L1 regularization term on weights]. Defaults to 0.

        clf_random_state (int, numpy.random.RandomState object, or None): [Random number seed. If int, this number is used to seed the C++ code. If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code. If None, default seeds in C++ code are used]. Defaults to None.

        silent (bool, optional): [Whether to print messages while running boosting]. Defaults to True.

        max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

        tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

        n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

        initial_strategy (str, optional): Which strategy to use for initializing the missing values in the training data (neighbor columns). IterativeImputer must initially impute the training data (neighbor columns) using a simple, quick imputation in order to predict the missing values for each target column. The ``initial_strategy`` argument specifies which method to use for this initial imputation. Valid options include: “most_frequent”, "populations", or "phylogeny". To inform the initial imputation for each sample, "most_frequent" uses the overall mode of each column, "populations" uses the mode per population with a population map file and the ``ImputeAlleleFreq`` class, and "phylogeny" uses an input phylogenetic tree and a rate matrix with the ``ImputePhylo`` class. Note that the "mean" and "median" options from the original IterativeImputer are not supported because they are not sensible settings for the type of input data used here. Defaults to "populations".

        str_encodings (dict(str: int), optional): [Integer encodings for nucleotides if input file was in STRUCTURE format. Only used if ``initial_strategy="phylogeny"``]. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}.

        imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

        skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

        random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

        verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
    """

    def __init__(
        self,
        genotype_data,
        *,
        prefix=None,
        gridparams=None,
        grid_iter=50,
        cv=5,
        validation_only=0.4,
        ga=False,
        population_size=10,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.1,
        ga_algorithm="eaMuPlusLambda",
        early_stop_gen=5,
        scoring_metric="accuracy",
        column_subset=0.1,
        chunk_size=1.0,
        disable_progressbar=False,
        progress_update_percent=None,
        n_jobs=1,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        boosting_type="gbdt",
        num_leaves=31,
        subsample_for_bin=200000,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_lambda=0,
        reg_alpha=0,
        clf_random_state=None,
        silent=True,
        n_nearest_features=10,
        max_iter=10,
        tol=1e-3,
        initial_strategy="most_frequent",
        str_encodings={"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        imputation_order="ascending",
        skip_complete=False,
        random_state=None,
        verbose=0,
    ):

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
    """[Impute missing data using a phylogenetic tree to inform the imputation]

    Args:
        genotype_data (GenotypeData object, optional): [GenotypeData object. If defined, some or all of the below optional are not required. If not defined, all the below options are required]. Defaults to None.

        alnfile (str, optional): [Path to PHYLIP or STRUCTURE-formatted file to impute]. Defaults to None.

        filetype (str, optional): [Filetype for the input alignment. Valid options include: "phylip", "structure1row", "structure1rowPopID", "structure2row", "structure2rowPopId". Not required if genotype_data is defined]. Defaults to "phylip".

        popmapfile (str, optional): [Path to population map file. Required if filetype is "phylip", "structure1row", or "structure2row". If filetype is "structure1rowPopID" or "structure2rowPopID", then the population IDs must be the second column of the STRUCTURE file. Not required if genotype_data is defined]. Defaults to None.

        treefile (str, optional): [Path to Newick-formatted phylogenetic tree file. Not required if genotype_data is defined with the guidetree option]. Defaults to None.

        qmatrix_iqtree (str, optional): [Path to *.iqtree file containing Rate Matrix Q table. If specified, ``ImputePhylo`` will read the Q matrix from the IQ-TREE output file. Cannot be used in conjunction with ``qmatrix`` argument. Not required if the ``qmatrix`` or ``qmatrix_iqtree`` options were used with the ``GenotypeData`` object]. Defaults to None.

        qmatrix (str, optional): [Path to file containing only a Rate Matrix Q table. Not required if genotype_data is defined with the qmatrix or qmatrix_iqtree option]. Defaults to None.

        str_encodings (dict(str: int), optional): [Integer encodings used in STRUCTURE-formatted file. Should be a dictionary with keys=nucleotides and values=integer encodings. The missing data encoding should also be included. Argument is ignored if using a PHYLIP-formatted file]. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}

        prefix (str, optional): [Prefix to use with output files]

        save_plots (bool, optional): [Whether to save PDF files with genotype imputations for each site. It makes one PDF file per locus, so if you have a lot of loci it will make a lot of PDF files]. Defaults to False.

        write_output (bool, optional): [Whether to save the imputed data to disk]. Defaults to True.

        disable_progressbar (bool, optional): [Whether to disable the progress bar during the imputation]. Defaults to False.

        kwargs (dict, optional): [Additional keyword arguments intended for internal purposes only. Possible arguments: {'column_subset': list(int) or np.ndarray(int)}; Subset SNPs by a list of indices]. Defauls to None.
    """

    def __init__(
        self,
        *,
        genotype_data=None,
        alnfile=None,
        filetype=None,
        popmapfile=None,
        treefile=None,
        qmatrix_iqtree=None,
        qmatrix=None,
        str_encodings={"A": 1, "C": 2, "G": 3, "T": 4, "N": -9},
        prefix="imputed_phylo",
        save_plots=False,
        write_output=True,
        disable_progressbar=False,
        **kwargs,
    ):
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

    def nbiallelic(self):
        """[Get the number of remaining bi-allelic sites after imputation]

        Returns:
            [int]: [Number of bi-allelic sites remaining after imputation]
        """
        return len(self.imputed.columns)

    def parse_arguments(self, genotype_data):
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

    def validate_arguments(self, genotype_data):
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

    def print_q(self, q):
        print("Rate matrix Q:")
        print("\tA\tC\tG\tT\t")
        for nuc1 in ["A", "C", "G", "T"]:
            print(nuc1, end="\t")
            for nuc2 in ["A", "C", "G", "T"]:
                print(q[nuc1][nuc2], end="\t")
            print("")

    def is_int(self, val):
        try:
            num = int(val)
        except ValueError:
            return False
        return True

    def impute_phylo(
        self, tree, genotypes, Q, site_rates=None, exclude_N=False
    ):
        """[Imputes genotype values by using a provided guide
        tree to inform the imputation, assuming maximum parsimony]

        Sketch:
            For each SNP:
            1) if site_rates, get site-transformated Q matrix

            2) Postorder traversal of tree to compute ancestral
            state likelihoods for internal nodes (tips -> root)
            If exclude_N==True, then ignore N tips for this step

            3) Preorder traversal of tree to populate missing genotypes
            with the maximum likelihood state (root -> tips)

        Args:
            tree ([toytree object]): [Toytree object]

            genotypes ([dict(list)]): [Dictionary with key=sampleids, value=sequences]

            Q ([pandas.DataFrame]): [Rate Matrix Q from .iqtree file]
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
                                genotypes[child.name][snp_index],
                                self.str_encodings,
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

    def get_nuc_colors(self, nucs):
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

    def label_bads(self, tips, labels, bads):
        for i, t in enumerate(tips):
            if t in bads:
                labels[i] = "*" + str(labels[i]) + "*"
        return labels

    def draw_imputed_position(self, tree, bads, genotypes, pos, out="tree.pdf"):
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
            # node_sizes = sizes,
            tip_labels=labels,
            width=400,
            height=600,
            **mystyle,
        )

        toyplot.pdf.render(canvas, out)

    def allMissing(self, tree, node_index, snp_index, genotypes):
        for des in tree.get_tip_labels(idx=node_index):
            if genotypes[des][snp_index].upper() not in ["N", "-"]:
                return False
        return True

    def get_internal_lik(self, pt, lik_arr):
        ret = list()
        for i, val in enumerate(lik_arr):

            col = list(pt.iloc[:, i])
            sum = 0.0
            for v in col:
                sum += v * val
            ret.append(sum)
        return ret

    def transition_probs(self, Q, t):
        ret = Q.copy(deep=True)
        m = Q.to_numpy()
        pt = scipy.linalg.expm(m * t)
        ret[:] = pt
        return ret

    def str2iupac(self, genotypes, str_encodings):
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

    def get_iupac_full(self, char, str_encodings):
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


class ImputeBackPropogation(GenotypeData):
    def __init__(
        self,
        genotype_data,
        *,
        num_reduced_dims=3,
        hidden_layers=3,
        hidden_layer_sizes=list(),
    ):

        super().__init__()

        encodings = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], -9: [0, 0, 0]}

        self.X = self.convert_onehot(
            genotype_data.genotypes_nparray, encodings_dict=encodings
        )

        # One for each one-hot encoded genotype
        self.Xref = self.X[:, :, 0]
        self.Xhet = self.X[:, :, 1]
        self.Xalt = self.X[:, :, 2]

        assert hidden_layers == len(hidden_layer_sizes) and hidden_layers > 0, (
            f"Hidden layers must be greater than 0 and of the same length as "
            f"hidden_layer_sizes, but got hidden_layers={hidden_layers} and "
            f"len(hidden_layer_sizes) == {len(hidden_layer_sizes)}"
        )

        # a = np.zeros((117, 2024))
        # b = np.ones((117, 2024))
        # c = np.full((117, 2024), 2)
        # test = np.stack((a, b, c), axis=2)
        # print(test)
        # print(test.shape)
        # sys.exit()

        self.Xt = None
        # self.invalid_mask = np.isnan(self.X)

        self.invalid_mask = np.where(
            genotype_data.genotypes_nparray == -9, True, False
        )

        self.valid_mask = np.where(~self.invalid_mask)
        self.l = hidden_layers
        self.V = np.random.randn(self.X.shape[0], num_reduced_dims)
        self.total_epochs = [0, 0, 0]
        self.x_r = T.vector()
        self.learning_rate = T.scalar("eta")
        self.c = T.iscalar()
        self.r = T.iscalar()
        self.gt = T.bscalar()

        self.V = theano.shared(
            np.array(
                np.random.rand(self.X.shape[0], num_reduced_dims),
                dtype=theano.config.floatX,
            )
        )

        self.weights = list()

        self.U = theano.shared(
            np.array(
                np.random.rand(num_reduced_dims, self.X.shape[1]),
                dtype=theano.config.floatX,
            )
        )

        self.single_layer = nnet.sigmoid(T.dot(self.U.T, self.V[self.r, :]))

        self.layers = list()

        for i in range(hidden_layers):
            if i == 0:
                self.weights.append(
                    initialize_weights(
                        (num_reduced_dims, hidden_layer_sizes[0])
                    )
                )

            else:
                self.weights.append(
                    initialize_weights(
                        (hidden_layer_sizes[i - 1], hidden_layer_sizes[i])
                    )
                )

        self.weights.append(
            initialize_weights((hidden_layer_sizes[-1], self.X.shape[1]))
        )

        for i in range(hidden_layers):
            if i == 0:
                self.layers.append(
                    nnet.sigmoid(T.dot(self.weights[i].T, self.V[self.r, :]))
                )

            else:
                self.layers.append(
                    nnet.sigmoid(T.dot(self.weights[i].T, self.layers[-1]))
                )

        self.layers.append(
            nnet.sigmoid(T.dot(self.weights[-1].T, self.layers[-1]))
        )

        self.fc1 = ((self.single_layer - self.x_r) ** 2)[self.c]
        self.fc = ((self.layers[-1] - self.x_r) ** 2)[self.c]

        self.phases = list()

        self.phases.append(
            theano.function(
                inputs=[
                    self.x_r,
                    self.r,
                    self.c,
                    theano.In(self.learning_rate, value=0.1),
                ],
                outputs=self.fc1,
                updates=[
                    (
                        self.U,
                        gradient_descent(self.fc1, self.U, self.learning_rate),
                    ),
                    (
                        self.V,
                        gradient_descent(self.fc1, self.V, self.learning_rate),
                    ),
                ],
            )
        )

        self.phases.append(
            theano.function(
                inputs=[
                    self.x_r,
                    self.r,
                    self.c,
                    theano.In(self.learning_rate, value=0.1),
                ],
                outputs=self.fc,
                updates=[
                    (
                        theta,
                        gradient_descent(self.fc, theta, self.learning_rate),
                    )
                    for theta in self.weights
                ],
            )
        )

        self.phases.append(
            theano.function(
                inputs=[
                    self.x_r,
                    self.r,
                    self.c,
                    theano.In(self.learning_rate, value=0.1),
                ],
                outputs=self.fc,
                updates=[
                    (
                        theta,
                        gradient_descent(self.fc, theta, self.learning_rate),
                    )
                    for theta in self.weights
                ]
                + [
                    (
                        self.V,
                        gradient_descent(self.fc, self.V, self.learning_rate),
                    )
                ],
            )
        )

        self.run_phase1 = theano.function(
            inputs=[self.r], outputs=self.single_layer
        )

        self.run = theano.function(inputs=[self.r], outputs=self.layers[-1])

    @timer
    def fit_predict(self):
        print("Doing Unsupervised Back-Propogation Imputation...")

        self.fit(phase=2)

        print(f"Initial REF RMSE: {self.get_rmse(self.Xt_ref, self.Xref)}")
        print(f"Initial HET RMSE: {self.get_rmse(self.Xt_het, self.Xhet)}")
        print(f"Initial ALT RMSE: {self.get_rmse(self.Xt_alt, self.Xalt)}")

        for i in range(1):
            print(f"Phase {(i + 1)}")

            self.initialize_params()

            for j in range(self.X.shape[2]):
                if j == 0:
                    dataset = "REF"
                elif j == 1:
                    dataset = "HET"
                elif j == 2:
                    dataset = "ALT"
                print(f"Predicting {dataset} dataset...")
                while self.current_eta[j] > self.target_eta:
                    self.s = self.train_epoch(phase=i)

                    if 1 - self.s[j] / self.s_[j] < self.gamma:
                        self.current_eta[j] = self.current_eta[j] / 2
                        print(f"Reduced eta to {self.current_eta[j]}")

                    self.s_[j] = self.s[j]
                    self.num_epochs[j] += 1
                    self.total_epochs[j] += 1
                    self.print_num_epochs(j)

    def fit(self, phase=2):
        # Initialize numpy array
        self.Xt_ref = np.zeros(self.Xref.shape)
        self.Xt_het = np.zeros(self.Xhet.shape)
        self.Xt_alt = np.zeros(self.Xalt.shape)

        if phase == 2 or phase == 3:
            for r in range(self.Xref.shape[0]):
                self.Xt_ref[r, :] = self.run(r)
                self.Xt_het[r, :] = self.run(r)
                self.Xt_alt[r, :] = self.run(r)

        elif phase == 1:
            for r in range(self.Xref.shape[0]):
                self.Xt_ref[r, :] = self.run_phase1(r)
                self.Xt_het[r, :] = self.run_phase1(r)
                self.Xt_alt[r, :] = self.run_phase1(r)

        else:
            raise Exception("Wrong phase provided!")

    def train_epoch(self, phase=1):
        start = default_timer()

        arr_rand = np.random.choice(
            len(self.valid_mask[0]), len(self.valid_mask[0]), replace=False
        )

        for r, c in zip(
            self.valid_mask[0][arr_rand], self.valid_mask[1][arr_rand]
        ):
            self.phases[phase](self.Xref[r, :], r, c, self.current_eta[0])
            self.phases[phase](self.Xhet[r, :], r, c, self.current_eta[1])
            self.phases[phase](self.Xalt[r, :], r, c, self.current_eta[2])

        end = default_timer()

        print(f"Epoch Training Time: {str((end-start) / 60)} minutes")

        self.fit()

        return [
            self.get_rmse(self.Xt_ref, self.Xref),
            self.get_rmse(self.Xt_het, self.Xhet),
            self.get_rmse(self.Xt_alt, self.Xalt),
        ]

    def print_num_epochs(self, idx, interval=10):
        if idx == 0:
            x = "REF"
        elif idx == 1:
            x = "HET"
        elif idx == 2:
            x = "ALT"

        if self.num_epochs[idx] % interval == 0:
            print(
                f"Epochs for Dataset {x}: {self.num_epochs[idx]}\tRMSE: {self.s[idx]}"
            )

    def initialize_params(self):
        self.initial_eta = [0.1, 0.1, 0.1]
        self.target_eta = 1e-4
        self.s = [0, 0, 0]
        self.s_ = [np.inf, np.inf, np.inf]
        self.current_eta = self.initial_eta
        self.gamma = 1e-5
        self.lambd = 1e-4
        self.num_epochs = [0, 0, 0]

    def get_rmse(self, arr_t, arr):
        return np.sqrt(
            np.mean((arr_t[~self.invalid_mask] - arr[~self.invalid_mask]) ** 2)
        )

    def predict(self):
        return np.stack((self.Xt_ref, self.Xt_het, self.Xt_alt), axis=2)


class ImputeAlleleFreq(GenotypeData):
    """
    [Impute missing data by global allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations]

    Args:
        genotype_data ([GenotypeData]): [GenotypeData instance. If ``genotype_data`` is not defined, then ``genotypes`` must be defined instead, and they cannot both be defined]. Defaults to None.

        by_populations (bool, optional): [Whether or not to impute by population or globally]. Defaults to False (globally).

        diploid (bool, optional): [When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency]. Defaults to True.

        default (int, optional): [Value to set if no alleles sampled at a locus]. Defaults to 0.

        missing (int, optional): [Missing data value]. Defaults to -9.


        prefix (str, optional): [Prefix for writing output files]

        write_output (bool, optional): [Whether to save imputed output to a file. If ``write_output`` is False, then just returns the imputed values as a pandas.DataFrame object. If ``write_output`` is True, then it saves the imputed data as a CSV file called ``<prefix>_imputed_012.csv``]

        output_format (str, optional): [Format of output imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list]. Defaults to "df".

        verbose (bool, optional): [Whether to print status updates. Set to False for no status updates]. Defaults to True.
    """

    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        by_populations=False,
        pops=None,
        diploid=True,
        default=0,
        missing=-9,
        prefix="out",
        write_output=True,
        output_format="df",
        verbose=True,
    ):

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

        self.imputed = self.fit_predict(gt_list)

        if write_output:
            self.write2file(self.imputed)

    def fit_predict(self, X):
        """[Impute missing genotypes using global allele frequencies, with missing alleles coded as negative; usually -9]

        Args:
            X ([list(list(int))]): [012-encoded genotypes obtained from the GenotypeData object as GenotypeData.genotypes_list]

        Returns:
            [pandas.DataFrame]: [Imputed genotypes of same dimensions as data]
        """
        if self.pops is not None and self.verbose:
            print("\nImputing by population allele frequencies...")
        elif self.pops is None and self.verbose:
            print("\nImputing by global allele frequency...")

        data = [item[:] for item in X]

        if self.pops is not None:
            pop_indices = misc.get_indices(self.pops)

        loc_index = 0
        for locus in data:
            if self.pops is None:
                allele_probs = self._get_allele_probs(locus, self.diploid)
                # print(allele_probs)
                if (
                    misc.all_zero(list(allele_probs.values()))
                    or not allele_probs
                ):
                    print(
                        "\nWarning: No alleles sampled at locus",
                        str(loc_index),
                        "setting all values to:",
                        str(self.default),
                    )
                    gen_index = 0
                    for geno in locus:
                        data[loc_index][gen_index] = self.default
                        gen_index += 1

                else:
                    gen_index = 0
                    for geno in locus:
                        if geno == self.missing:
                            data[loc_index][gen_index] = self._sample_allele(
                                allele_probs, diploid=True
                            )
                        gen_index += 1

            else:
                for pop in pop_indices.keys():
                    allele_probs = self._get_allele_probs(
                        locus,
                        self.diploid,
                        missing=self.missing,
                        indices=pop_indices[pop],
                    )

                    if (
                        misc.all_zero(list(allele_probs.values()))
                        or not allele_probs
                    ):
                        print(
                            "\nWarning: No alleles sampled at locus",
                            str(loc_index),
                            "setting all values to:",
                            str(self.default),
                        )
                        gen_index = 0
                        for geno in locus:
                            data[loc_index][gen_index] = self.default
                            gen_index += 1
                    else:
                        gen_index = 0
                        for geno in locus:
                            if geno == self.missing:
                                data[loc_index][
                                    gen_index
                                ] = self._sample_allele(
                                    allele_probs, diploid=True
                                )
                            gen_index += 1

            loc_index += 1

        if self.verbose:
            print("Done!")

        if self.output_format == "df":
            return pd.DataFrame(data)

        elif self.output_format == "array":
            return np.array(data)

        elif self.output_format == "list":
            return data

        else:
            raise ValueError("Unknown output output_format specified")

    def _sample_allele(self, allele_probs, diploid=True):
        if diploid:
            alleles = misc.weighted_draw(allele_probs, 2)
            if alleles[0] == alleles[1]:
                return alleles[0]
            else:
                return 1
        else:
            return misc.weighted_draw(allele_probs, 1)[0]

    def _get_allele_probs(
        self, genotypes, diploid=True, missing=-9, indices=None
    ):
        data = genotypes
        length = len(genotypes)

        if indices is not None:
            data = [genotypes[index] for index in indices]
            length = len(data)

        if len(set(data)) == 1:
            if data[0] == missing:
                ret = dict()
                return ret
            else:
                ret = dict()
                ret[data[0]] = 1.0
                return ret

        if diploid:
            length = length * 2
            ret = {0: 0.0, 2: 0.0}
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
                    print(
                        "\nWarning: Ignoring unrecognized allele",
                        str(g),
                        "in get_allele_probs\n",
                    )
            for allele in ret.keys():
                ret[allele] = ret[allele] / float(length)
            return ret
        else:
            ret = dict()
            for key in set(data):
                if key != missing:
                    ret[key] = 0.0
            for g in data:
                if g == missing:
                    length -= 1
                else:
                    ret[g] += 1
            for allele in ret.keys():
                ret[allele] = ret[allele] / float(length)
            return ret

    def get_imputed_genotypes(self):
        """[Getter for the imputed genotypes]

        Returns:
            [pandas.DataFrame]: [Imputed 012-encoded genotypes as DataFrame]
        """
        return self.imputed

    def write2file(self, df):
        outfile = f"{self.prefix}_imputed_012.csv"
        df.to_csv(outfile, header=False, index=False)
