Tutorial
========

PG-SUI Overview
________________

PG-SUI (Population Genomic Supervised and Unsupervised Imputation) performs missing data imputation on SNP datasets. We have included seven machine and deep learning algorithms with which to impute, including both supervised and unsupervised training methods. The currently supported algorithms include:

+ Supervised
    + XGBoost
    + RandomForest
    + K-Nearest Neighbors

+ The supervised algorithms work by using the N nearest features, based on absolute correlations between loci, to supervise the imputation. The algorithm is based on the MICE (Multivariate Imputation by Chained Equations) algorithm implemented in scikit-learn's IterativeImputer ([1]_). The unsupervised deep learning models each have distinct architectures that perform the imputation in different ways. For training the deep learning algorithms, missing values are simulated and the model is trained on the simulated missing values. The real missing values are then predicted by the trained model. The strategy for simulating missing values can be set with the ``sim_strategy`` argument.

+ Unsupervised Neural networks
    + Variational AutoEncoder (VAE) [2]_
    + Standard AutoEncoder (SAE) [3]_
    + Non-linear Principal Component Analysis (NLPCA) [4]_
    + Unsupervised backpropagation (UBP) [5]_

+ NLPCA
    + NLPCA trains randomly generated, reduced-dimensionality input to predict the correct output. The input is then refined at each backpropagation step until it accurately predict the output.
+ UBP
    + UBP is an extension of NLPCA with the input being randomly generated and of reduced dimensionality that gets trained to predict the supplied output based on only known values. It then uses the trained model to predict missing values. However, in contrast to NLPCA, UBP trains the model over three phases. The first is a single layer perceptron used to refine the randomly generated input. The second phase is a multi-layer perceptron that uses the refined reduced-dimension data from the first phase as input. In the second phase, the model weights are refined but not the input. In the third phase, the model weights and the inputs are then refined.


+ Standard AutoEncoder
    + The SAE model reduces (i.e., encodes) the input, which is the full dataset, to a reduced-dimensional layer, and then trains itself to reconstruct the input (i.e., decodes) from the reduced-dimensional layer. 

+ VAE
    + The VAE model is similar to SAE, except the reduced-dimensional layer (latent dimension) represents a sampling distribution with a mean and variance that gets sampled from, and the model trains itself by trying to reconstruct the input (i.e., decodes).


Installing PG-SUI
------------------

Use pip to install PG-SUI:

.. code-block:: bash

    pip install pgsui

Input Data
-----------

To use PG-SUI, you need to load your data into the API using the GenotypeData class from the `SNPio package <https://github.com/btmartin721/SNPio>`_, which is automatically installed as a dependency through pip. GenotypeData takes a STRUCTURE, VCF, or PHYLIP file, a population map (popmap) file, and optionally phylogenetic tree and rate matrix files as input. See the `SNPio documentation <https://snpio.readthedocs.io>`_ for more information.

Here is a basic usage for SNPio:

.. code-block:: python

    from snpio import GenotypeData

    data = GenotypeData(
        filename="pgsui/example_data/phylip_files/test_n100.phy",
        popmapfile="pgsui/example_data/popmaps/test.popmap",
        guidetree="pgsui/example_data/trees/test.tre",
        qmatrix="pgsui/example_data/trees/test.qmat",
        siterates="pgsui/example_data/trees/test_siterates_n100.txt",
        prefix="test_imputer",
        force_popmap=True,
        plot_format="png",
    )


Supported Imputation Methods
----------------------------

There are numerous supported algorithms to impute missing data. Each one can be run by calling the corresponding class and specifying the necessary parameters and settings that you want to change from default. All of them require the GenotypeData object as input.

.. code-block:: python

    from pgsui import *

    # Various imputation methods are supported

    ############################################
    # Supervised IterativeImputer classifiers
    ############################################

    knn = ImputeKNN(genotype_data=data) # K-Nearest Neighbors
    rf = ImputeRandomForest(genotype_data=data) # Random Forest or Extra Trees
    xgb = ImputeXGBoost(genotype_data=data) # XGBoost

    ########################################
    # Non-machine learning methods
    ########################################

    # Use phylogeny to inform imputation
    phylo = ImputePhylo(genotype_data=data)

    # Use by-population or global allele frequency to inform imputation
    pop_af = ImputeAlleleFreq(genotype_data=data, by_populations=True)
    global_af = ImputeAlleleFreq(genotype_data=data, by_populations=False)

    # Matrix factorization imputation
    mf = ImputeMF(genotype_data=data)

    ########################################
    # Unsupervised neural network models
    ########################################

    vae = ImputeVAE(genotype_data=data) # Variational autoencoder
    sae = ImputeStandardAutoEncoder(genotype_data=data) # Standard AutoEncoder
    nlpca = ImputeNLPCA(genotype_data=data) # Nonlinear PCA
    ubp = ImputeUBP(genotype_data=data) # Unsupervised backpropagation

In each of the above class instantiations, the analysis will automatically run. Each method has its own unique arguments, so look over :doc:`API documentation <pgsui.impute>` to see what each of the parameters do.

The imputed data will be saved as a GenotypeData object that can be accessed from the ``imputed`` property of the class instance. For example:

.. code-block:: python

    vae = ImputeVAE(genotype_data=data)

    # Get the new GentoypeData instance.
    imputed_genotype_data = vae.imputed


Initial Strategy
----------------

For the supervised IterativeImputer method, the ``initial_strategy`` argument determines the initial method for imputing the nearest neighbors that are used to inform the column currently being imputed. There are several options you can choose from for ``initial_strategy``. "populations" uses the popmap file to inform the imputation. "most_frequent" uses the global mode per column, and "phylogeny" uses an input phylogeny. "mf" uses matrix factorization to do the initial imputation. 

Different options might be better or worse, depending on the dataset. It helps to know some biological context of your study system in this case. For example, you can use a phylogenetic tree to do the initial imputation in the supervised models and to inform the missing data simulations in the neural network models.

.. code-block:: python

    xgb_data = ImputeXGBoost(genotype_data=data, initial_strategy="phylogeny")
    nlpca_data = ImputeNLPCA(genotype_data=data, sim_strategy="nonrandom")

.. note::

    If using ``initial_strategy="phylogeny"``, then you must input a phylogeny when initializing the ``GenotypeData`` object. 
    
    Likewise, if using ``initial_strategy="populations"``, then a popmap file must be supplied to ``GenotypeData``.  

Nearest Neighbors, Iterations, and Estimators
---------------------------------------------

N-Nearest Neighbors
^^^^^^^^^^^^^^^^^^^

IterativeImputer uses the N-nearest neighbors (columns) based on a correlation matrix. The number of nearest neighbors can be tuned by changing the ``n_nearest_features`` parameter.

.. code-block:: python

    lgbm = ImputeXGBoost(genotype_data=data, n_nearest_features=50)

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

The IterativeImputer algorithms support dataset chunking. If you find yourself running out of RAM, try breaking the imputation into chunks.

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

IterativeImputer is a `scikit-learn <https://scikit-learn.org>`_ imputation method that we have extended herein. It iterates over each feature (i.e., SNP column) and uses the N-nearest neighbors to inform the imputation at the current feature. The number of nearest neighbors (i.e., columns) can be adjusted by users, and neighbors are determined using a correlation matrix between features.

IterativeImputer can use any of scikit-learn's estimators, but currently PG-SUI supports Random Forest (or Extra Trees), XGBoost, and K-Nearest Neighbors.

Our modifications have added grid searches and some other customizations to scikit-learn's `IterativeImputer class <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>`_.


Parallel Processing
-------------------

The IterativeImputer classifiers have an ``n_jobs`` parameter that tell it to parallelize the estimator. If ``gridparams`` is not None, ``n_jobs`` is used for the grid search. Otherwise it is used for the classifier. -1 means using all available processors.

The neural network classifiers use all processors by default, but if ``gridparams`` is not None, then it uses n_jobs to parallelize parameter sweeps in the grid search.

.. code-block:: python

    # Use all available CPU cores.
    rf = ImputeRandomForest(genotype_data=data, n_jobs=-1)

    # Use 4 CPU cores.
    rf = ImputeRandomForest(genotype_data=data, n_jobs=4)


Imputer validation
------------------

Both IterativeImputer and the neural networks calculate a suite of validation metrics to assess the efficacy of the model and facilitate cross-comparison. For IterativeImputer, there are two ways to validate: Parameter grid searches and cross-validation replicates. The validation runs on a random subset of the SNP columns, the proportion of which can be changed with the ``column_subset`` argument. If you want to do the validation, set ``do_validation=True``.

E.g.,:

.. code-block:: python

    # Do validation on a random subset of 25% of the columns.
    rf = ImputeRandomForest(genotype_data=data, do_validation=True, column_subset=0.25)


Grid searches
^^^^^^^^^^^^^

The IterativeImputer methods can perform several types of grid searches by providing the ``gridparams`` argument. Grid searches try to find the best combinations of parameters by maximizing the accuracy across a distribution of parameter values. If ``gridparams=None``, the grid search will not be performed. If ``gridparams != None:``, the grid search will run.

Three types of grid searches can be run:
    1. GridSearchCV: Tests all provided parameter combinations supplied in ``gridparams``.
    2. RandomizedSearchCV: Generates random parameters from a distribution or a list/ array of provided values. The number of parameter combinations to test can be set with the ``grid_iter`` parameter.
    3. Genetic Algorithm: Use a genetic algorithm to refine the grid search. It tries to optimize the search space with the genetic algorithm. Will also generate several informative plots.

    The type of grid search can be set with the ``gridsearch_method`` argument to the estimator, which supports the following options: ``gridsearch``, ``randomized_gridsearch``, and ``genetic_algorithm``.


.. warning::

    GridSearchCV tests every possible combination of model parameters. So, if you supply a lot of parameter possibilities it will take a really long time to run. The number of parameters combinations contains ``C = L1 x L2 x L3 x ... x Ln`` possible combinations, where each ``L`` is the length of the list for a given parameter.

.. note::
    RandomizedSearchCV tests ``grid_iter * cv`` random parameter combinations. So, if you are doing 5-fold cross-validation and you have 1000 parameter combinations, it will test 5000 total folds.

.. note::
    See the scikit-learn `model selection documentation <https://scikit-learn.org/stable/model_selection.html>`_ for more information on GridSearchCV and RandomizedSearchCV.

The genetic algorithm has a suite of parameters that can be adjusted. See the :doc:`documentation <pgsui.impute>` and `the sklearn-genetic-opt documentation <https://sklearn-genetic-opt.readthedocs.io/en/stable/api/space.html>`_ for more information.


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
        gridsearch_method="gridsearch",
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

Then you can run the grid search in the same way, except set ``gridsearch_method=genetic_algorithm``.

.. code-block:: python

    # Use 25% of columns to do Genetic Algorithm grid search.
    rf = ImputeRandomForest(
        genotype_data=data, 
        gridparams=grid_params, 
        column_subset=0.25, 
        gridsearch_method="genetic_algorithm",
    )

You can change how many cross-validation folds the grid search uses by setting the ``cv`` parameter.

.. code-block:: python

    rf = ImputeRandomForest(genotype_data=data, cv=3)

Cross-validation
^^^^^^^^^^^^^^^^

If you don't want to do a grid search and just want to do cross-validation, then you can just leave the default ``gridparams=None`` and set ``do_validation`` to True. 

.. code-block:: python

    # Use 25% of columns to do cross-validation without grid search.
    rf = ImputeRandomForest(
        genotype_data=data, 
        column_subset=0.25,
        do_validation=True
    )

Or you can do the imputation without any validation metrics.

.. code-block:: python

    # No validation
    rf = ImputeRandomForest(
        genotype_data=data, 
        do_validation=False, # default
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

This will run it with the default arguments. You might want to adjust some of the parameters. See the relevant :doc:`documentation <pgsui.impute>` for more information.

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

.. tip:: Recommended Usage

Our recommendation for the neural networks is to start with the grid searches and to maximize the roc_auc scores or other any other metrics of your choice (see the `scikit-learn metrics documentation <https://scikit-learn.org/stable/modules/model_evaluation.html>`_).


Non-ML Imputers
---------------

We also have classes to impute using non-machine learning methods. You can impute by the global or by-population mode per column, using an input phylogeny to inform the imputation, and by matrix factorization. We also have the ``ImputeRefAllele`` imputer that will always just set missing values to the reference allele. These methods can be used both as the ``initial_strategy`` with IterativeImputer and the neural networks and as standalone imputation methods.

Impute by Allele Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we impute by global allele frequency:

.. code-block:: python

    # Global allele frequency per column
    global_af = ImputeAlleleFreq(
        genotype_data=data, 
        by_populations=False,
    )

And we can impute with the by-population mode like this:

.. code-block:: python

    pop_af = ImputeAlleleFreq(
        genotype_data=data, 
        by_populations=True,
    )

Alternatively, we can just have it impute by the reference allele in all cases:

.. code-block:: python

    ref_af = ImputeRefAllele(genotype_data=data)

Impute with Phylogeny
^^^^^^^^^^^^^^^^^^^^^

We can also use a phylogeny to inform the imputation. In this case, we would have had to specify the Newick-formatted tree file and the Rate Matrix Q to the ``GenotypeData`` object first.

.. code-block:: python

    # Popmap file used
    # Structure file in 2-row per individual format
    data = GenotypeData(
        filename="pgsui/example_data/phylip_files/test_n100.phy",
        popmapfile="pgsui/example_data/popmaps/test.popmap",
        guidetree="pgsui/example_data/trees/test.tre",
        qmatrix="pgsui/example_data/trees/test.qmat",
        siterates="pgsui/example_data/trees/test_siterates_n100.txt",
        prefix="test_imputer",
        force_popmap=True,
        plot_format="png",
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


References
-----------

.. [1] Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.

.. [2] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.

.. [3] Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

.. [4] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.

.. [5] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.
