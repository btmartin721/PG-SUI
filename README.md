# Super Deli  

![Super Deli Logo](./logos/full_logo_1280x640.png.png?raw=true "Logo")

A comprehensive machine learning species delimitation package

## Input files

Takes a structure or phylip file and a popmap file as input.  
There are a number of options for the structure file format. See the help menu:

```python super_deli.py -h```  

```
To-Do's:

- Get data in a format so that dimensionality reduction can be done.
- read_vcf
- aligning across K algorithms 
- genotype imputation (i.e. by a priori population and globally) -- as method to GenotypeData 
- DelimitationModel Class with each UML method class inheriting? Or something like that.

Notes:
- Can calculate AIC for k-means, or other clustering methods? See https://stackoverflow.com/questions/15839774/how-to-calculate-bic-for-k-means-clustering-in-r#:~:text=The%20AIC%20can%20be%20calculated,2*m*k)%20%7D 

Dimension reduction
1. Using scikit-allel to do PCA
2. cMDS
3. isoMDS
4. DAPC -- with a priori groups, or groups from other algorithms, or an analog of find.clusters (=kmeans)
5. Others?

Clustering algorithms
1. K-Means
2. Hierarchical clustering
3. DBSCAN
4. PAM
5. Others?

Decision Trees
1. Random Forest
2. XGBOOST: Extreme Gradient Boosting

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

Give users a multitude of statistics
```  

