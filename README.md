# Super Deli  

![Super Deli Logo](https://github.com/btmartin721/super_deli/blob/master/logos/full_logo_1280x640.png)

A comprehensive machine learning species delimitation package

## Requirements

+ python == 3.7
+ pandas >= 1.2.5
+ numpy >= 1.20
+ matplotlib
+ seaborn
+ kneed
+ tqdm
+ jupyterlab
+ scikit-learn >= 0.24
+ scikit-learn-extra
+ sklearn-genetic-opt >= 0.6.0
+ toytree
+ scipy >= 1.6.2 and <= 1.7.0

Python versions other than 3.7 are not currently supported.  

The requirements can mostly be installed with conda. The only module that isn't available on conda is sklearn-genetic-opt, which can be installed via pip.

```
create -n super_deli python=3.7
conda activate super_deli
conda install matplotlib seaborn jupyterlab scikit-learn tqdm pandas=1.2.5 numpy=1.20.2 scipy=1.6.2
conda install -c conda-forge toytree scikit-learn-extra kneed
pip install sklearn-genetic-opt[all] # For genetic algorithm plotting functions
```

## Input files

Takes a structure or phylip file and a popmap file as input.  
There are a number of options for the structure file format. See the help menu:

```python super_deli.py -h```  

## API Pipeline (so far)

```
# Read in PHYLIP or STRUCTURE-formatted file
# and impute missing data
# Various imputation options are supported thus far
data = GenotypeData(...)
data.impute_missing(...)

# Initialize DimReduction object
# Instance used for most functions downstream
dr = DimReduction(data.imputed_df, ...)

# Run Principal Component Analysis
# Can be used as input for runRandomForestUML()
pca = runPCA(dr, ...)
pca.plot() # makes a scatterplot

# Random Forest unsupervised
rf = runRandomForestUML(dr, ...)

# runMDS can be used with random forest dissimilarity matrix (recommended),
# which is accessible as a runRandomForest class property like below.
rf_cmds = runMDS(dr, metric=True, dissimilarity_matrix=rf.dissimilarity_matrix, ...)

# Or it can be run on the raw RF output like this:
rf_cmds = runMDS(dr, metric=True, rf=rf.rf_model, ...)

# Then it can be plotted by calling plot:
rf_cmds.plot()

PAM clustering can then be run on the MDS output:
cmds_pam = PamClustering(rf_cmds, dr, ...)

# And mean silhouette widths can be used to determine PAM optimal K
cmds_pam.msw(plot_msw=True)

# If you don't want to plot it, set plot_msw=False.
# In that case it will just get the average silhouette scores for each K.
# If plot_msw=True, it will also get average silhouette scores plus make
# some neat plots.

# isoMDS can be run by setting metric=False in runMDS()
rf_isomds = runMDS(dr, metric=False, ...)
rf_isomds.plot()

isomds_pam = PamClustering(rf_cmds, dr, ...)
isomds_pam.msw(...)

# t-SNE can be run in the same way
tsne = runTSNE(dr, ...)
tsne.plot()

tsne_pam = PamClustering(tsne, dr, ...)
tsne_pam.msw(...)
```

## To-Dos


To-Do's:

- Get data in a format so that dimensionality reduction can be done.
- read_vcf
- aligning across K algorithms 
- genotype imputation (i.e. by a priori population and globally) -- as method to GenotypeData 
- DelimitationModel Class with each UML method class inheriting? Or something like that.

Notes:
- Can calculate AIC for k-means, or other clustering methods? See https://stackoverflow.com/questions/15839774/how-to-calculate-bic-for-k-means-clustering-in-r#:~:text=The%20AIC%20can%20be%20calculated,2*m*k)%20%7D 

Dimension reduction  
1. ~~Using scikit-allel to do PCA~~  
2. ~~cMDS~~  
3. ~~isoMDS~~  
4. DAPC -- with a priori groups, or groups from other algorithms, or an analog of find.clusters (=kmeans)
5. Others?

Clustering algorithms
1. K-Means
2. Hierarchical clustering
3. DBSCAN  
4. ~~PAM~~  
5. Others?

Decision Trees  
1. ~~Random Forest~~  

Deep Learning
1. Variational autoencoders
2. Convolutional neural networks
3. Others?

Accuracy plotting and comparisons
1. Plot accuracy and model loss
2. AUC plots
3. Comparing across algorithms and models
4. Allow users to compare everything to assess what's best for their dataset

Plot replicates and align K across replicates
1. Aligning K across replicates
2. Stacked barplots

Use unsupervised learning to predict labels for supervised algorithms
1. XGBoost
2. Logistic Regression
3. Bayesian Regression
4. Linear Discriminant Analysis
5. K-Nearest Neighbors
6. Stochastic Gradient Descent
7. LightGBM
8. Naive Bayes
9. AdaBoost
10. Random Forests 

Semi-supervised learning  

~~Phylogenetic trees to impute missing data~~  

Phylogenetic trees to supervise machine learning analyses


Give users a multitude of statistics
