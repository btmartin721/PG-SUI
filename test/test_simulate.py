import unittest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import msprime

from pgsui.simulators.simulate import SNPulatoRate, SNPulator, SNPulatorConfig
from snpio import GenotypeData
import matplotlib.pylab as plt
import numpy as np


import numpy as np


def check_data_integrity(data):
    """
    Check the integrity of a 2D list or array.

    Args:
        data (list of list or np.ndarray): A 2D list or array.

    Returns:
        str: A message indicating the integrity of the data.
    """
    # Convert to NumPy array for easier manipulation
    np_data = np.array(data)

    # Check if all rows have the same number of columns
    shape = np_data.shape
    if len(set(len(row) for row in data)) != 1:
        return "Irregular shape: Not all rows have the same number of columns."

    # Check for mixed data types
    dtypes = set(str(np_data.dtype) for row in np_data)
    if len(dtypes) > 1:
        return "Mixed data types detected."

    return f"Data integrity looks good. Shape: {shape}, Data type: {list(dtypes)[0]}"


def convert_012(alignment):
    """
    Convert the most frequent genotype in each column of an alignment to 0,
    IUPAC ambiguity codes to 1, and the variant allele(s) to 2.

    Args:
        alignment (list of list): A 2D list of shape (n_samples, n_columns) representing the alignment.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_samples, n_columns) with converted genotypes.
    """
    # Define IUPAC ambiguity codes
    iupac_ambiguity_codes = set("RYSWKMBDHV")
    alignment_array = np.array(alignment)

    # Convert the input list to a NumPy array for efficient operations
    alignment_array = np.array(alignment)

    # Initialize an empty array to store the converted genotypes
    converted_array = np.zeros_like(alignment_array, dtype=int)

    # Loop through each column to find the most frequent genotype, ambiguity codes, and variant alleles
    for col_idx in range(alignment_array.shape[1]):
        column = alignment_array[:, col_idx]

        # Find the most frequent genotype in the column
        unique_elements, counts = np.unique(column, return_counts=True)
        most_frequent_genotype = unique_elements[np.argmax(counts)]

        # Loop through each element in the column to convert it
        for row_idx, genotype in enumerate(column):
            if genotype == most_frequent_genotype:
                converted_array[row_idx, col_idx] = 0
            elif genotype in iupac_ambiguity_codes:
                converted_array[row_idx, col_idx] = 1
            else:
                converted_array[row_idx, col_idx] = 2
    return converted_array


def plot_class_distribution(y_masked, filename="class_distribution.png"):
    """
    Plot and save the distribution of classes 0, 1, and 2 in the masked true labels.

    Args:
        y_masked (np.ndarray): Masked true labels, a 1D numpy array.
        filename (str): Filename to save the plot.

    Returns:
        None: The function saves the plot to disk.
    """

    # Count the occurrences of each class
    unique_elements, counts_elements = np.unique(y_masked, return_counts=True)

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(unique_elements, counts_elements, color=["red", "green", "blue"])

    # Annotate the bars with the actual counts
    for i, count in enumerate(counts_elements):
        plt.text(
            unique_elements[i], count, str(count), ha="center", va="bottom"
        )

    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title("Distribution of Classes in Masked y_true")

    # Save the plot
    # plt.savefig(filename)
    plt.show()


class TestMutationRateCalculations(unittest.TestCase):
    def setUp(self):
        # Create a MultipleSeqAlignment object with sequences of length 20
        # seq1 = SeqRecord(Seq("ACTGACTGACTGACTGACTG"), id="seq1")
        # seq2 = SeqRecord(Seq("ACTGACTAACTGACTGACTA"), id="seq2")
        # seq3 = SeqRecord(Seq("ACTAACTGACTGACTAACTG"), id="seq3")
        # seq4 = SeqRecord(Seq("ACTGACCCACTGACTGACCC"), id="seq4")
        # seq5 = SeqRecord(Seq("ACTGGCTGACTGACTGGCTG"), id="seq5")
        # self.alignment = MultipleSeqAlignment([seq1, seq2, seq3, seq4, seq5])

        self.genotype_data = GenotypeData(
            filename="PG-SUI/pgsui/example_data/phylip_files/test.phy",
            popmapfile="PG-SUI/pgsui/example_data/popmaps/test.popmap",
            force_popmap=True,
        )

        self.alignment = self.genotype_data.alignment

        config = {
            "demes_graph": "PG-SUI/pgsui/example_data/demography/example_demes2.yaml",
            "mutation_rate": 1e-5,
            "sequence_length": 500,
            "recombination_rate": 1e-2,
            "record_migrations": False,
            "mutation_model": msprime.GTR,
        }

        self.snp_config = config

    def test_kappa(self):
        snprate = SNPulatoRate(self.genotype_data)
        calculated_kappa = snprate._calculate_kappa(self.alignment)
        print(f"Calculated Kappa: {calculated_kappa}\n\n")

    def test_base_frequencies(self):
        snprate = SNPulatoRate(self.genotype_data)
        calculated_base_frequencies = snprate._calculate_base_frequencies()
        print(
            f"Calculated Base Frequencies: {calculated_base_frequencies}\n\n"
        )

    def test_simulate(self):
        snpconfig = SNPulatorConfig(**self.snp_config)
        # snprate = SNPulatoRate(self.genotype_data)
        # snpconfig.update_mutation_rate(snprate.calculate_rate(model="GTR"))
        snprate = SNPulatoRate(self.genotype_data, time=1e7)
        rate = snprate.calculate_rate(model="GTR")
        snpconfig.update_mutation_rate(rate)
        print(f"Estimated mutation rate: {snpconfig.mutation_rate}\n\n")

        snp = SNPulator(self.genotype_data, snpconfig)
        new_samples, new_pops = snp.sim_ancestry(
            sample_sizes=[60, 60], populations=["pop1", "pop2"]
        )
        snp.sim_mutations()
        snp_data = snp.genotypes

        snp_data2 = snp.simulate_snp(100, 2000)
        print(snp_data2)

        self.genotype_data.write_phylip(
            "PG-SUI/pgsui/example_data/phylip_files/simtest1.phy",
            snp_data=snp_data,
            samples=new_samples,
            verbose=True,
        )

        with open(
            "PG-SUI/pgsui/example_data/popmaps/simtest1.popmap", "w"
        ) as fout:
            for s, p in zip(new_samples, new_pops):
                fout.write(f"{s}\t{p}\n")

        self.genotype_data_sim = GenotypeData(
            filename="PG-SUI/pgsui/example_data/phylip_files/simtest1.phy",
            popmapfile="PG-SUI/pgsui/example_data/popmaps/simtest1.popmap",
        )

        gt_012 = self.genotype_data_sim.genotypes_012(fmt="numpy")
        plot_class_distribution(gt_012.ravel())


if __name__ == "__main__":
    unittest.main()
