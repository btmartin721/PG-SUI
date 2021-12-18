Tutorial
========

Input Data
----------

To use PG-SUI, you need to load your data into the API using the GenotypeData class. GenotypeData takes a STRUCTURE or PHYLIP file, the file type, a population map (popmap) file, and optionally phylogenetic tree and rate matrix files as input.

Popmap File
^^^^^^^^^^^

The population map is a two-column, tab-separated file in the format:

.. code-block:: text

    sample1    population1
    sample2    population1
    sample3    population2
    sample4    population2
    ...        ...

PHYLIP Input
^^^^^^^^^^^^

If you choose to load a PHYLIP file, you also need to specify a population map.

To load the PHYLIP file, instantiate the GenotypeData class.

.. code-block:: python

    data = GenotypeData(
        filename="test.phy", 
        filetype="phylip", 
        popmapfile="test.popmap"
    )

STRUCTURE File
^^^^^^^^^^^^^^

To load a STRUCTURE file, there are a few options you can choose from. The file can be in either the 1-row or 2-row per individual format by setting ``filetype=structure1row`` or ``filetype=structure2row``, and the population IDs can either be included as the second column or specified with the popmap file.


.. code-block:: python

    # Popmap file used
    # Structure file in 2-row per individual format
    data = GenotypeData(
        filename="test.nopops.str",
        filetype="structure2row",
        popmapfile="test.popmap"
    )

Load a Phylogeny
^^^^^^^^^^^^^^^^

Some of the analyses require you to input a Newick-formatted phylogenetic tree and a substitution rate matrix Q.

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

GenotypeData Attributes
^^^^^^^^^^^^^^^^^^^^^^^

The data can be retrieved from the GenotypeData object as a pandas DataFrame, a 2D numpy array, or a 2D list, each with shape (n_samples, n_SNPs):

.. code-block:: python

    df = data.genotypes_df # pandas DataFrame
    arr = data.genotypes_nparray # numpy array
    l = data.genotypes_list # python list

You can also retrieve the number of individuals and SNP sites:

.. code-block:: python

    num_inds = data.indcount
    num_snps = data.snpcount

And to retrieve a list of sample IDs or population IDs:

.. code-block:: python

    inds = data.individuals
    pops = data.populations


Supported Imputation Methods
----------------------------

There are numerous supported algorithms to impute missing data. Each one can be run by calling the corresponding class and specifying the necessary parameters and settings that you want to change from default. All of them require the GenotypeData object as input.

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

In each of the above class instantiations, the analysis will automatically run. Each method has its own unique arguments, so look over the :doc:`API documentation <pgsui.impute.estimators>` to see what each of the parameters do.

The imputed data will be written to a file on disk with the prefix designated by the ``prefix`` parameter, or you can access the imputed data from the instantiated object. Some of the imputers have options to turn off writing to disk.

Initial Strategy
----------------

For the IterativeImputer method, the ``initial_strategy`` argument determines the initial method for imputing the nearest neighbors that are used to inform the column currently being imputed. There are several options you can choose from for ``initial_strategy``. "populations" uses the popmap file to inform the imputation. "most_frequent" uses the global mode per column, and "phylogeny" uses an input phylogeny. "mf" uses matrix factorization to do the initial imputation. 

Both the IterativeImputer and the neural network methods use the ``initial_strategy`` argument for doing the validation.

Different options might be better or worse, depending on the dataset. It helps to know some biological context of your study system in this case.

.. code-block:: python

    ImputeXGBoost(genotype_data=data, initial_strategy="phylogeny")

.. note::

    If using ``initial_strategy="phylogeny"``, then you must input a phylogeny when initializing the ``GenotypeData`` object. 
    
    Likewise, if using ``initial_strategy="populations"``, then a popmap file must be supplied to ``GenotypeData``.

Nearest Neighbors, Iterations, and Estimators
---------------------------------------------

N Nearest Neighbors
^^^^^^^^^^^^^^^^^^^

IterativeImputer uses the N nearest neighbors (columns) based on a correlation matrix. The number of nearest neighbors can be tuned by changing the ``n_nearest_features`` parameter.

.. code-block:: python

    lgbm = ImputeLGBM(genotype_data=data, n_nearest_features=50)

Maximum Iterations
^^^^^^^^^^^^^^^^^^

Likewise, IterativeImputer will make up to ``max_iter`` passes through the columns to assess convergence. This value can be changed if the passes are not converging. Note that there is an early stopping criterion implemented, so if they converge early the imputation will stop early.

.. code-block:: python

    knn = ImputeKNN(genotype_data=data, max_iter=50)

Number of Estimators
^^^^^^^^^^^^^^^^^^^^

The decision tree classifiers also have an ``n_estimators`` parameter that can be adjusted. Increasing ``n_estimators`` can make the model better at the expense of computational resources.

.. code-block:: python

    rf = ImputeRandomForest(genotype_data=data, n_estimators=200)

.. warning::

    Setting n_nearest_features and n_estimators too high can lead to extremely high resource usage and long run times.

Chunk size
----------

Both the IterativeImputer and neural network algorithms support dataset chunking. If you find yourself running out of RAM, try breaking the imputation into chunks.

.. code-block:: python

    # Split dataset into 25% chunks.
    rf = ImputeRandomForest(
        genotype_data=data, 
        max_iter=50, 
        n_estimators=200, 
        n_nearest_features=30,
        chunk_size=0.25
    )

Progress Bar
------------

If you are working on your own local machine, you can use the fancy TQDM progress bar that we have implemented. But if you are working on a distributed environment such as a high performance computing cluster, you might need to turn off the TQDM progress bar if it is not working correctly. We provide an option to do so in all the models.

.. code-block:: python

    rf = ImputeRandomForest(genotype_data=data, disable_progressbar=True)

It will still print status updates to the screen, it just won't use the TQDM progress bar.

If you disable the progress bar and want to change how often it prints status updates, you can do so with the ``progress_update_percent`` option.

.. code-block:: python

    # Print status updates after every 20% completed.
    rf = ImputeRandomForest(
        genotype_data=data, 
        disable_progressbar=True, 
        progress_update_percent=20
    )

Iterative Imputer
-----------------

IterativeImputer is a `scikit-learn <https://scikit-learn.org>`_ imputation method that we have extended herein. It iterates over each feature (i.e., SNP column) and uses the N nearest neighbors to inform the imputation at the current feature. The number of nearest neighbors (i.e., features) can be adjusted by users, and neighbors are determined using a correlation matrix between features.

IterativeImputer can use any of scikit-learn's estimators, but currently PG-SUI supports Random Forest (or Extra Trees), Gradient Boosting, K-Nearest Neighbors, XGBoost, LightGBM, 

Our modifications have added grid searches and some other customizations to scikit-learn's `IterativeImputer class <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`_.


Parallel Processing
-------------------

Many of the IterativeImputer classifiers have an ``n_jobs`` parameter that tell it to paralellize the estimator. If ``gridparams`` is not None, ``n_jobs`` is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors.

.. code-block:: python

    # Use all available CPU cores.
    rf = ImputeRandomForest(genotype_data=data, n_jobs=-1)

    # Use 4 CPU cores.
    rf = ImputeRandomForest(genotype_data=data, n_jobs=4)


Imputer validation
------------------

Both IterativeImputer and the neural networks calculate a suite of validation metrics to assess the efficacy of the model and facilitate cross-comparison. For IterativeImputer, there are two ways to validate: Parameter grid searches and cross-validation replicates. For the neural network models, just the cross-validation replicates are performed. The validation runs on a random subset of the SNP columns, the proportion of which can be changed with the ``column_subset`` (for grid searches) and ``validation_only`` (for cross-validation) arguments.

E.g.,:

.. code-block:: python

    rf = ImputeRandomForest(genotype_data=data, validation_only=0.25)


Grid searches
^^^^^^^^^^^^^

The IterativeImputer methods can perform several types of grid searches by providing the ``gridparams`` argument. Grid searches try to find the best combinations of settings by maximizing the accuracy across a distribution of parameter values. If ``gridparams == None``, the grid search will not be performed and just a cross-validation will be performed by running a user-specified number of imputation replicates and calculating validation metrics. If ``gridparams != None:``, the grid search will run. If ``gridparams == None`` and ``validation_only == None``, then no validation will be performed.

Two types of grid searches can be run:

    1. RandomizedSearchCV: Generates random parameters from a distribution or a list/ array of provided values.
    2. Genetic Algorithm: Use a genetic algorithm to refine the grid search. Will generate several informative plots.

The genetic algorithm has a suite of parameters that can be adjusted. See the :doc:`documentation <pgsui.impute.estimators>` and `the sklearn-genetic-opt documentation <https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html>`_ for more information.


gridparams
""""""""""

The gridparams argument is a dictionary with the keys as the parameter keywords and the values a list, array, or distribution to sample from. What you provide to ``gridparams`` are the parameters that will be involved in the grid search. Unprovided parameters will not undergo the grid search.

If using RandomizedSearchCV, it should be similar to the following. The arguments will change depending which classifier is being used. The following are arguments for ``ImputeRandomForest()``:

.. code-block:: python

    # For RandomizedSearchcv
    # Number of features to consider at every split
    max_features = ["sqrt", "log2"]

    # Maximum number of levels in the tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimmum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(2, 10, num=5)]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(1, 5, num=5)]

    # Make the gridparams object:
    grid_params = {
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }

Then you would run the analysis by providing the gridparams argument. 

.. code-block:: python

    # Use 25% of columns to do RandomizedSearchCV grid search.
    rf = ImputeRandomForest(
        genotype_data=data, 
        gridparams=grid_params, 
        column_subset=0.25, 
        ga=False
    )

To run the genetic algorithm grid search, the parameter distributions need to be set up using the sklearn-genetic-opt API instead of lists/ arrays. You can use the ``Categorical``, ``Integer``, and ``Continuous`` classes to set up the distributions (see the `sklearn-genetic-opt documentation <https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html>`_)

.. code-block:: python

    # Genetic Algorithm grid_params
    grid_params = {
        "max_features": Categorical(["sqrt", "log2"]),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 10),
        "max_depth": Integer(2, 110),
    }

Then you can run the grid search in the same way, except set ``ga=True``.

.. code-block:: python

    # Use 25% of columns to do Genetic Algorithm grid search.
    rf = ImputeRandomForest(
        genotype_data=data, 
        gridparams=grid_params, 
        column_subset=0.25, 
        ga=True
    )

You can change how many cross-validation folds the grid search uses by setting the ``cv`` parameter.

.. code-block:: python

    rf = ImputeRandomForest(genotype_data=data, cv=3)

Cross-validation
^^^^^^^^^^^^^^^^

If you don't want to do a grid search and just want to do cross-validation, then you can just leave the default ``gridparams=None``.

.. code-block:: python

    # Use 25% of columns to do cross-validation without grid search.
    rf = ImputeRandomForest(
        genotype_data=data, 
        validation_only=0.25
    )

Or you can do the imputation without any validation metrics.

.. code-block:: python

    # No validation
    rf = ImputeRandomForest(
        genotype_data=data, 
        validation_only=None
    )

You can change the number of replicates that it does by setting the ``cv`` parameter.

.. code-block:: python

    rf = ImputeRandomForest(genotype_data=data, cv=3)

.. note::

    The ``cv`` parameter functions differently when using grid searches versus doing the validation replicates. For grid searches, it does stratified K folds and performs cross-validation to estimate the accuracy. 
    
    For doing the validation replicates, ``cv`` is used to set the number of replicates that are performed. The evalutation metrics are then reported as the average (for numeric parameters) or mode (for categorical parameters) of the replicates.


Neural Network Imputers
-----------------------

The neural network imputers can be run in the same way with cross-validation.

.. code-block:: python

    nlpca = ImputeNLPCA(genotype_data=data)

This will run it with the default arguments. You might want to adjust some of the parameters. See the relevant :doc:`documentation <pgsui.impute.estimators>` for more information.

The neural network methods print out the current mean squared error with each epoch (cycle through the data). The VAE model will run for a fixed number of epochs, but the NLPCA and UBP models have an early stopping criterion that will checkpoint the model at the first occurrence of the lowest error and stop training after a lack of improvement for a user-defined number of epochs. This is intended to reduce overfitting.

If you find that the model is not converging or is converging very slowly, try adjusting the ``learning_rate`` parameter. Lowering it will slow down convergence, but if the error is fluctuating a lot lowering ``learning_rate`` can prevent that from happening. Alternatively, if the model is converging super slowly, you can try increasing ``learning_rate``.

.. code-block:: python

    # Lower the learning_rate parameter.
    ImputeNLPCA(genotype_data=data, learning_rate=0.01)

You might also want to experiment with the number of hidden layers or the size of the hidden layers. Hidden layers allow the neural network to learn non-linear patterns, and you can try adjusting the ``num_hidden_layers`` and ``hidden_layer_sizes`` parameters. ``hidden_layer_sizes`` supports a list of integers of the same length as ``num_hidden_layers``, or you can specify a string to get the midpoint ("midpoint"), square root ("sqrt"), or natural logarithm ("log2") of the total number of columns.

.. code-block:: python

    nlpca = ImputeNLPCA(genotype_data=data, num_hidden_layers=2, hidden_layer_sizes="sqrt")

You should also experiment with the ``hidden_activation``, ``batch_size``, and ``train_epochs`` (for VAE) parameters. If your accuracy is low, adjusting these can help, and for VAE if the error converges far earlier than training ends, overfitting can occur and the ``train_epochs`` parameter should be reduced.

.. code-block:: python

    vae = ImputeVAE(genotype_data=data, hidden_activation="elu", batch_size=64, train_epochs=50)

    nlpca=ImputeNLPCA(genotype_data=data, hidden_activation="relu", batch_size=64)

See the `keras documentation <https://www.tensorflow.org/api_docs/python/tf/keras/activations>`_ for more information on the supported hidden activation functions.

Finally, for NLPCA and UBP you can experiment with the number of reduced-dimensional components. Usually, 2 or 3 dimensions is a good rule of thumb.

.. code-block:: python

    ubp = ImputeUBP(genotype_data=data, n_components=2)

Our recommendation for the neural networks is to try to maximize the accuracy and other metrics. So you will likely need to run it several times and adjust the parameters.


Non-ML Imputers
---------------

We also have classes to impute using non-machine learning methods. You can impute by the global or by-population mode per column, using an input phylogeny to inform the imputation, and by matrix factorization. These methods can be used both as the ``initial_strategy`` with IterativeImputer and the neural networks and as standalone imputation methods.

Impute by Allele Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we impute by global allele frequency:

.. code-block:: python

    # Global allele frequency per column
    global_af = ImputeAlleleFreq(
        genotype_data=data, 
        by_populations=False,
        write_output=True
    )

And we can impute with the by-population mode like this:

.. code-block:: python

    pop_af = ImputeAlleleFreq(
        genotype_data=data, 
        by_populations=True,
        pops=data.populations, 
        write_output=True
    )

Impute with Phylogeny
^^^^^^^^^^^^^^^^^^^^^

We can also use a phylogeny to inform the imputation. In this case, we would have had to specify the Newick-formatted tree file and the Rate Matrix Q to the ``GenotypeData`` object first.

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

    phy = ImputePhylo(genotype_data=data, write_output=True)

You can also save a phylogeny plot per site with the known and imputed values as the tip labels.

.. code-block:: python

    phy = ImputePhylo(genotype_data=data, write_output=True, save_plots=True)

.. warning::

    This will save one plot per SNP column, so if you have hundreds or thousands of loci, it will output hundreds or thousands of PDF files.

Matrix Factorization
^^^^^^^^^^^^^^^^^^^^

Finally, you can impute using matrix factorization:

.. code-block:: python

    ImputeMF(genotype_data=data)

