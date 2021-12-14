Input Data
==========

To use PG-SUI, you need to load your data into the API using the GenotypeData class. GenotypeData takes a STRUCTURE or PHYLIP file, the file type, a population map (popmap) file, and optionally phylogenetic tree and rate matrix files as input.

Popmap File
===========

The population map is a two-column, tab-separated file in the format:

.. code-block:: text

    sample1    population1
    sample2    population1
    sample3    population2
    sample4    population2
    ...        ...

PHYLIP Input
============

If you choose to load a PHYLIP file, you also need to specify a population map.

To load the PHYLIP file, instantiate the GenotypeData class.

.. code-block:: python

    data = GenotypeData(
        filename="test.phy", 
        filetype="phylip", 
        popmapfile="test.popmap"
    )

To load a STRUCTURE file, there are a few options you can choose from. The file can be in either the 1-row or 2-row per individual format by setting ``filetype=structure1row`` or ``filetype=structure2row``, and the population IDs can either be included as the second column or specified with the popmap file.

.. code-block:: python

    # Popmap file used
    # Structure file in 2-row per individual format
    data = GenotypeData(
        filename="test.nopops.str",
        filetype="structure2row",
        popmapfile="test.popmap"
    )

To initialize the GenotypeData object with a phylogeny, which is required if using the ``initial_strategy="phylogeny`` argument or ``ImputePhylo()`` method, you need to specify paths to the Newick-formatted tree file and the rate matrix Q file.

The rate matrix Q file can be obtained from the IQ-TREE standard output, and in fact you can just input the whole .iqtree file and PG-SUI will pull the rate matrix from it. Or you can provide a file that only has the rate matrix table.

.. code-block:: python

    # Popmap file used
    # Structure file in 2-row per individual format
    data = GenotypeData(
        filename="test.nopops.str",
        filetype="structure2row",
        popmapfile="test.popmap",
        guidetree="test.tre",
        qmatrix_iqtree="test.iqtree",
        # qmatrix="test.Q" # Instead of IQ-TREE file
    )


The data can be retrieved from the GenotypeData object as a pandas DataFrame, a 2D numpy array, or a 2D list, each with shape (n_samples, n_SNPs):

.. code-block:: python

    df = data.genotypes_df
    arr = data.genotypes_nparray
    l = data.genotypes_list

You can also retrieve the number of individuals and SNP sites:

.. code-block:: python

    num_inds = data.indcount
    num_snps = data.snpcount

And to retrieve a list of sample IDs or population IDs:

.. code-block:: python

    inds = data.individuals
    pops = data.populations


Supported Imputation Methods
============================

There are numerous supported algorithms to impute missing data. Each one can be run by calling the corresponding class and specifying the necessary parameters and options that you want to change from default. All of them require the GenotypeData object as input.

.. code-block:: python

    # Various imputation methods are supported

    ############################################
    # Supervised IterativeImputer classifiers
    ############################################

    knn = ImputeKNN(genotype_data=data) # K-Nearest Neighbors
    rf = ImputeRandomForest(genotype_data=data) # Random Forest or Extra Trees
    gb = ImputeGradientBoosting(genotype_data=data) # Gradient Boosting
    xgb = ImputeXGBoost(genotype_data=data) # XGBoost
    lgbm = ImputeLightGBM(genotype_data=data) # LightGBM

    ########################################
    # Non-machine learning methods
    ########################################

    # Use phylogeny to inform imputation
    phylo = ImputePhylo(genotype_data=data)

    # Use by-population or global allele frequency to inform imputation
    pop_af = ImputeAlleleFreq(genotype_data=data, by_populations=True)
    global_af = ImputeAlleleFreq(genotype_data=data, by_populations=False)

    mf = ImputeMF(genotype_data=data) # Matrix factorization

    ########################################
    # Unsupervised neural network models
    ########################################

    vae = ImputeVAE(genotype_data=data) # Variational autoencoder
    nlpca = ImputeNLPCA(genotype_data=data) # Nonlinear PCA
    ubp = ImputeUBP(genotype_data=data) # Unsupervised backpropagation

In each of the above class instantiations, the analysis will automatically run. Each method has its own unique arguments, so look over the :doc:`API documentation <pgsui>` to see what each of the parameters do.

Imputer validation
==================

Cross-validation
----------------

All methods calculate several validation metrics to assess the efficacy of the model. The validation runs on a random subset of the SNP columns, the proportion of which can be changed with the ``validation_only`` argument. E.g.,:

.. code-block:: python

    rf = ImputeRandomForest(genotype_data=data, validation_only=0.25)

Grid searches
-------------

The IterativeImputer methods can also perform several types of grid searches by providing the ``gridparams`` argument. Grid searches try to find the best combinations of settings by maximizing the accuracy. If ``gridparams=None``, the grid search will not be performed and a cross-validation will be performed by running a user-specified number of imputation replicates. If ``gridparams!=None:``, the grid search will run. If ``gridparams=None`` and ``validation_only=None``, then no validation will be performed.

E.g.,:

.. code-block:: python

Two types of grid searches can be run:

    1. RandomizedSearchCV: Generates random parameters from a distribution.
    2. Genetic Algorithm: Use a genetic algorithm to refine the grid search. Will generate several nice plots.

The gridparams argument is a dictionary with the keys as the keyword settings and the values a list or distribution to sample from. What you provide to ``gridparams`` are the parameters involved in the grid search.

If using RandomizedSearchCV, it should be similar to the following. The arguments will change depending on the classifier being used. The following are arguments for ``ImputeRandomForest()``:

.. code-block:: python

    # For RandomizedSearchcv
    # Number of trees in random forest
    n_estimators = [
        int(x) for x in np.linspace(start=100, stop=1000, num=10)
    ]

    # Number of features to consider at every split
    max_features = ["sqrt", "log2"]

    # Maximum number of levels in the tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimmum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(2, 10, num=5)]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(1, 5, num=5)]

    # Proportion of dataset to use with bootstrapping
    # max_samples = [x for x in np.linspace(0.5, 1.0, num=6)]

    # # Random Forest gridparams - RandomizedSearchCV
    grid_params = {
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }

