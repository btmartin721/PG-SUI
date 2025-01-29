About PG-SUI
============

PG-SUI: Population Genomic Supervised and Unsupervised Imputation

PG-SUI Philosophy
-----------------

PG-SUI is a cutting-edge Python 3 API designed to impute missing values from population genomic SNP datasets using advanced supervised and unsupervised machine learning and deep learning algorithms. The package also includes non-machine learning imputation methods, providing flexibility and robust solutions for handling missing data in genomic SNP datasets.

Supervised Imputation Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supervised methods utilize state-of-the-art machine learning approaches to perform imputation. The centerpiece is scikit-learn's IterativeImputer, which applies the MICE (Multivariate Imputation by Chained Equations) algorithm [1]_ to iteratively impute each SNP feature based on the relationships with other features. PG-SUI extends this functionality by allowing the use of the following classifiers to inform imputation:
	•	K-Nearest Neighbors
	•	Random Forest
	•	Extra Trees
	•	XGBoost

Users can customize the number of nearest features (neighbors) considered during imputation. The flexibility of these models allows researchers to tailor their approach to the specific properties of their genomic data.

See the scikit-learn documentation <https://scikit-learn.org>_ for additional details on IterativeImputer and the classifiers supported.

Unsupervised Imputation Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unsupervised methods in PG-SUI include a suite of custom neural network models optimized for population genomic data:
	•	Variational Autoencoder (VAE) [2]_
	•	Standard Autoencoder (SAE) [3]_
	•	Non-linear Principal Component Analysis (NLPCA) [4]_
	•	Unsupervised Backpropagation (UBP) [5]_

These methods leverage deep learning to infer missing genotypes by learning patterns and relationships within the data:
	1.	Training on Simulated Missing Data: Real missing values are masked, and simulated missing values are generated during training. Models are trained only on known values to reconstruct complete genotypes.
	2.	Prediction of Real Missing Values: After training, the models predict the actual missing values using the learned patterns.

Details on Neural Network Imputation
	•	SAE (Standard Autoencoder): Compresses loci into a reduced-dimensional latent space (e.g., 2-3 dimensions), then reconstructs the original genotype data through a decoder network.
	•	VAE (Variational Autoencoder): Extends SAE by learning a latent distribution (mean and variance) in the reduced-dimensional space. During training, samples from this distribution are passed into the decoder for reconstruction, ensuring a robust probabilistic framework.
	•	NLPCA (Non-linear PCA): Begins with randomly initialized reduced-dimensional inputs and refines them iteratively by minimizing reconstruction error for known genotypes. NLPCA functions similarly to the decoder phase of an autoencoder.
	•	UBP (Unsupervised Backpropagation): Builds upon NLPCA with a three-phase training process.

        1.	Phase 1: Refines reduced-dimensional input with a single-layer neural network.
        2.	Phase 2: Refines network weights using a multi-layer perceptron (MLP).
        3.	Phase 3: Refines both the reduced-dimensional input and MLP weights simultaneously, achieving high imputation accuracy.

Non-Machine Learning Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PG-SUI also incorporates several non-machine learning approaches to provide robust and interpretable baseline solutions for imputation:
	•	Per-population mode per SNP site: The most frequent allele in each population is used to impute missing genotypes.
	•	Global mode per SNP site: The most frequent allele across the entire dataset is used.
	•	Reference allele per SNP site: Imputes missing genotypes using the reference allele.
	•	Phylogeny-informed Imputation: Utilizes an input phylogeny to guide imputation, accounting for evolutionary relationships.
	•	Matrix Factorization: A matrix decomposition-based method for predicting missing values.

These non-machine learning methods can serve as:

	•	Standalone imputers.
	•	Initial imputation strategies for IterativeImputer.
	•	Validation benchmarks for both machine learning and deep learning models.

Why Choose PG-SUI?
^^^^^^^^^^^^^^^^^^

PG-SUI is a flexible, efficient, and extensible package that combines classical statistical approaches with cutting-edge machine learning and deep learning models to address missing data in population genomic SNP datasets. Researchers can select from a wide variety of supervised, unsupervised, and non-machine learning methods to best suit their needs.

References

.. [1] Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.

.. [2] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.

.. [3] Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

.. [4] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.

.. [5] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.