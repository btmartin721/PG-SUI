Simulation Strategies
======================

This document describes the different strategies implemented in the `SimGenotypeDataTransformer` class. This class is designed to simulate missing values over known values to evaluate how well they are predicted by various deep learning models implemented in PyTorch.

The `SimGenotypeDataTransformer` class supports several strategies for simulating missing data. Below is a description of each strategy and how to use them:

1. **Random Strategy**:
    - `strategy="random"`
    - Randomly masks a specified proportion of known genotype calls.

2. **Random Balanced Strategy**:
    - `strategy="random_balanced"`
    - Masks genotype calls with balanced proportions across different genotypes (0, 1, 2).

3. **Random Inverse Strategy**:
    - `strategy="random_inv"`
    - Masks genotype calls inversely proportional to their frequency.

4. **Random Balanced Multinomial Strategy**:
    - `strategy="random_balanced_multinom"`
    - Uses a multinomial distribution to mask genotype calls with balanced proportions.

5. **Random Inverse Multinomial Strategy**:
    - `strategy="random_inv_multinom"`
    - Uses a multinomial distribution to mask genotype calls inversely proportional to their frequency.

6. **Nonrandom Strategy**:
    - `strategy="nonrandom"`
    - Uses a phylogenetic tree to simulate missing data in a nonrandom manner.

7. **Nonrandom Weighted Strategy**:
    - `strategy="nonrandom_weighted"`
    - Uses a phylogenetic tree with genotype-based weighting to simulate missing data.

8. **Nonrandom Distance Strategy**:
    - `strategy="nonrandom_distance"`
    - Uses a distance-based approach to simulate non-random missingness.

Each strategy can be customized further by adjusting parameters such as `prop_missing`, `missing_val`, `mask_missing`, and `seed` to control the proportion of missing data, the value used for missing data, whether to mask existing missing data, and the random seed for reproducibility.

The following code snippet demonstrates how to use the `SimGenotypeDataTransformer` class with the random strategy:

.. code-block:: python

    from snpio import GenotypeEncoder, VCFReader
    from pgsui import ImputeAutoencoder

    # Load a VCF file and encode the genotypes
    genotype_data = VCFReader("example.vcf", popmapfile="example.popmap")

    # Encode the genotypes using the GenotypeEncoder class
    ge = GenotypeEncoder(genotype_data)

    # Use the 'random_inv_multinom' strategy to simulate the missing values.
    ae = ImputeAutoencoder(genotype_data, sim_prop_missing=0.3, sim_strategy="random_inv_multinom")

    imputed_data = ae.fit_transform(ge.genotypes_012)

For more information on the `SimGenotypeDataTransformer` class and its parameters, please refer to the API documentation.