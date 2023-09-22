import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import log
from pathlib import Path
from typing import Optional, Union
import inspect

import demes
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
import toytree as tt
import tskit
from pgsui.utils.sequence_tools import get_iupac_codes
from scipy.optimize import curve_fit

from pgsui.utils.misc import get_int_iupac_dict


class SNPulatorUtils:
    @staticmethod
    def validate_positive_number(value, name="Value", min=0):
        if value < min:
            raise ValueError(f"{name} must be a positive number > {min}.")

    @staticmethod
    def validate_mutation_model(value, name="Value"):
        instance_models = [
            msprime.GTR,
            msprime.HKY,
            msprime.F84,
            msprime.SLiMMutationModel,
        ]

        str_models = ["jc69", "binary", "infinite_alleles", "blosum62", "pam"]
        instance_models = [msprime.HKY, msprime.F84, msprime.GTR, msprime.JC69]

        if isinstance(value, str):
            if value.lower() not in str_models:
                raise ValueError(
                    f"{name} must be either a string in {str_models} or an "
                    f"instance of {instance_models}"
                )
        else:
            if value not in instance_models:
                raise ValueError(
                    f"{name} must be either a string in {str_models} or an "
                    f"instance of {instance_models}"
                )

    @staticmethod
    def validate_demographic_model(model):
        """Validate msprime demographic model."""
        return model.validate()

    @staticmethod
    def validate_isfile(value, name="Value"):
        if not Path(value).is_file():
            raise FileNotFoundError(f"{name} file could not be found.")

    @staticmethod
    def validate_isdir(value, name="Value"):
        if not Path(value).is_dir():
            raise OSError(f"{name} directory does not exist.")

    @staticmethod
    def validate_type(value, type, name="Value"):
        if not isinstance(value, type):
            raise TypeError(
                f"{name} must be of type {type}, but got {type(value)}"
            )

    @staticmethod
    def validate_arglist(*args, max=1):
        check = [a for a in args if not a or a is None]
        if sum(x is True for x in check) > max:
            raise ValueError(f"Only one of {args} can be set at a time.")

    @staticmethod
    def validate_num_range(value, min=0, max=1, name="Value"):
        if value < min:
            raise ValueError(
                f"{name} must be in the range {min, max}, but got {value}"
            )
        if value > max:
            raise ValueError(
                f"{name} must be in the range {min, max}, but got {value}"
            )

    @staticmethod
    def validate_genotype_data_popmap(genotype_data):
        if (
            not hasattr(genotype_data, "popmap_inverse")
            or genotype_data.popmap_inverse is None
        ):
            raise AttributeError(
                "A popmap file was not supplied to the GenotypeData object. A "
                "popmap is required to use the msprime alignment simulations."
            )

    @staticmethod
    def check_sum_to_1(value, name="Value"):
        if isinstance(value, dict):
            value_sum = sum(value.values())
        elif isinstance(value, list):
            value_sum = sum(value)
        else:
            raise TypeError(
                f"{name} must be a dict or list, but got {type(value)}"
            )

        if value_sum != 1:
            raise ValueError(
                f"{name} values must sum to 1, but got {value_sum}"
            )


class SNPulatoRate:
    def __init__(self, genotype_data, time=None, bootstrap=1000):
        """
        Initialize the SNPulatorRate class.

        Args:
            genotype_data (snpio.GenotypeData): GenotypeData instance.
            time (float, optional): Time since divergence, for normalizing mutation rate. Can be either in years or generations. If provided, the mutation rate will be normalized with the divergence time. Defaults to None.
            bootstrap (int): Number of bootstrap iterations for confidence intervals (default is 1000).
        """
        self.genotype_data = genotype_data.copy()
        self.time = time
        self.bootstrap = bootstrap
        self.alignment = genotype_data.alignment
        self.num_sequences = len(self.alignment)
        self.seq_length = self.alignment.get_alignment_length()
        self.base_freq = self._calculate_base_frequencies()

    def _normalize_base_frequencies(self, base_freq):
        """Normalize the base frequencies to sum to 1."""
        total = sum(base_freq.values())
        return {base: freq / total for base, freq in base_freq.items()}

    def _calculate_base_frequencies(self):
        """
        Calculate the frequencies of each base (A, C, G, T) in the alignment.

        Returns:
            dict: Base frequencies for A, C, G, T.
        """
        base_count = Counter()
        total_bases = 0

        for i in range(self.num_sequences):
            for base in self.alignment[i].seq:
                if base not in {"-", "N", "?", "."}:  # Ignore gaps
                    base_count[base] += 1
                    total_bases += 1

        base_freq = {
            base: count / total_bases
            for base, count in base_count.items()
            if base in {"A", "C", "G", "T"}
        }

        # Ensure all bases are present in the dictionary,
        # even if their frequency is zero
        for base in ["A", "C", "G", "T"]:
            if base not in base_freq:
                base_freq[base] = 0.0

        # Sort the dictionary by key before sending.
        return self._normalize_base_frequencies(
            dict(sorted(base_freq.items()))
        )

    def _rescale_rate_matrix(self, rate_matrix, base_frequencies):
        """
        Rescale the rate matrix by dividing all elements by the average rate of substitution.

        Args:
            rate_matrix (np.ndarray): The original rate matrix.
            base_frequencies (dict): Dictionary of base frequencies for each valid base.

        Returns:
            np.ndarray: The rescaled rate matrix.
        """
        # Calculate the average rate of substitution (mu)
        valid_bases = list(base_frequencies.keys())
        mu = 0.0
        for i, base1 in enumerate(valid_bases):
            for j, base2 in enumerate(valid_bases):
                if base1 != base2:
                    freq1 = base_frequencies[base1]
                    rate = rate_matrix[i, j]
                    mu += freq1 * rate

        # Rescale the rate matrix by dividing all elements by mu
        rate_matrix_rescaled = rate_matrix / mu

        return rate_matrix_rescaled

    def _calculate_mu(self, valid_bases={"A", "C", "G", "T"}):
        """
        Estimate the overall rate of substitution (mu) based on a given sequence alignment.

        Args:
            valid_bases (set): Set of valid bases to consider, defaults to {'A', 'C', 'G', 'T'}.

        Returns:
            float: Estimated overall rate of substitution (mu).
        """
        alignment = self.alignment

        num_substitutions = 0
        total_sites = 0

        num_sequences = len(alignment)
        seq_length = alignment.get_alignment_length()

        # Count the number of substitutions and the total number of valid sites
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                for pos in range(seq_length):
                    base1, base2 = alignment[i, pos], alignment[j, pos]
                    if base1 in valid_bases and base2 in valid_bases:
                        total_sites += 2  # Increment by 2 to account for both sequences in the pair
                        if base1 != base2:
                            num_substitutions += 1

        # Calculate the overall rate of substitution
        mu = num_substitutions / total_sites if total_sites > 0 else 0

        return mu

    @staticmethod
    def _calculate_kappa(alignment, valid_bases={"A", "C", "G", "T"}):
        """
        Calculate the transition/transversion rate ratio (kappa) from a sequence alignment,
        with debugging information.

        Args:
            valid_bases (set): Set of valid bases to consider, defaults to {'A', 'C', 'G', 'T'}.

        Returns:
            float: The calculated transition/transversion rate ratio (kappa).
        """
        alignment = np.array(alignment)
        global_transition_counter = 0
        global_transversion_counter = 0

        # Define transitions and transversions
        transitions = [{"A", "G"}, {"C", "T"}]

        # Create a mask for valid bases
        valid_mask = np.isin(alignment, list(valid_bases))

        # Loop through each pair of sequences in the alignment
        num_sequences, seq_length = alignment.shape
        for i in range(num_sequences):
            for j in range(i + 1, num_sequences):
                transition_counter = 0  # Reset for each pair
                pair_mask = valid_mask[i] & valid_mask[j]
                bases_i = alignment[i, pair_mask]
                bases_j = alignment[j, pair_mask]

                # Count transitions and transversions
                for base1, base2 in zip(bases_i, bases_j):
                    if base1 != base2:
                        if {base1, base2} in transitions:
                            transition_counter += 1
                        else:
                            global_transversion_counter += 1

                global_transition_counter += transition_counter

        # Calculate kappa
        kappa = (
            global_transition_counter / global_transversion_counter
            if global_transversion_counter > 0
            else float("inf")
        )

        print(kappa)
        return kappa

    class GTR:
        def __init__(self, alignment, base_freq: dict, time: int) -> None:
            self.alignment = alignment
            self.base_freq = base_freq
            self.time = time

            self._rate = self._calculate_gtr_rate()

        def _calculate_relative_rates(self):
            """
            Calculate the 6 unique relative rates (R) for the GTR model based on a given alignment and base frequencies.

            Args:
                None: Uses class attributes for alignment and base frequencies.

            Returns:
                list: The 6 unique relative rates (R) between each pair of nucleotides.
            """
            # Step 1: Data Preparation
            alignment = self.alignment
            base_freq = self.base_freq

            valid_bases = ["A", "C", "G", "T"]
            base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
            count_matrix = np.zeros((4, 4))

            for i in range(len(alignment)):
                for j in range(i + 1, len(alignment)):
                    for pos in range(len(alignment[0])):
                        base1, base2 = alignment[i][pos], alignment[j][pos]

                        if (
                            base1 not in valid_bases
                            or base2 not in valid_bases
                        ):
                            continue

                        count_matrix[
                            base_to_index[base1], base_to_index[base2]
                        ] += 1
                        count_matrix[
                            base_to_index[base2], base_to_index[base1]
                        ] += 1  # It's reversible

            # Step 2: Normalization
            total_count = np.sum(count_matrix) - np.sum(np.diag(count_matrix))
            p_matrix = count_matrix / total_count

            # Step 3: Calculate R
            r_values = {}

            for i, base1 in enumerate(valid_bases):
                for j, base2 in enumerate(valid_bases):
                    if i < j:
                        r_ij = p_matrix[i, j] / (
                            base_freq[base1] * base_freq[base2]
                        )
                        r_values[f"{base1}{base2}"] = r_ij

            # Convert dict_values to a list of floats
            relative_rates = list(map(float, r_values.values()))
            return np.array(relative_rates, dtype=float)

        def _calculate_gtr_rate(self):
            """
            GTR rate calculation.

            Returns:
                float: Calculated GTR mutation rate.
            """
            rate_matrix = self._calculate_gtr_Q()
            base_freq = self.base_freq
            # rate_matrix = self._rescale_rate_matrix(rate_matrix, base_freq)

            # Ensure the order of bases in rate_matrix and base_freq is
            # consistent
            valid_bases = list(base_freq.keys())

            total_mutations = 0
            num_sequences = len(self.alignment)
            seq_length = self.alignment.get_alignment_length()

            # Loop through each pair of sequences
            for i in range(num_sequences):
                for j in range(i + 1, num_sequences):
                    # Loop through each position in the sequences
                    for pos in range(seq_length):
                        base1, base2 = (
                            valid_bases[pos % 4],
                            valid_bases[(pos + 1) % 4],
                        )  # Simulated bases

                        i, j = valid_bases.index(base1), valid_bases.index(
                            base2
                        )
                        rate = rate_matrix[i, j]

                        # Incorporate base frequencies
                        freq1 = base_freq.get(base1, 0)
                        freq2 = base_freq.get(base2, 0)
                        scaled_rate = rate * freq1 * freq2
                        total_mutations += scaled_rate

            # Calculate the mutation rate
            mutation_rate = total_mutations / (
                seq_length * (num_sequences * (num_sequences - 1)) / 2
            )

            if self.time:
                mutation_rate /= self.time

            return mutation_rate

        def _calculate_gtr_Q(self):
            """
            Calculate and normalize the GTR (General Time Reversible) model rate matrix (Q) from a given sequence alignment.

            Args:

                alignment (MultipleSeqAlignment): Sequence alignment.

                valid_bases (set): Set of valid bases to consider, defaults to {'A', 'C', 'G', 'T'}.

                normalize (bool): Whether to normalize the rate matrix.

            Returns:
                np.ndarray: The calculated and possibly normalized rate matrix.

            """
            gtr = msprime.GTR(
                relative_rates=self._calculate_relative_rates(),
                equilibrium_frequencies=list(self.base_freq.values()),
            )
            return gtr.transition_matrix

        @property
        def rate(self):
            SNPulatorUtils.validate_positive_number(self._rate)
            return self._rate

    class JC69:
        def __init__(self, alignment, base_freq: dict, time: int) -> None:
            self.alignment = alignment
            self.base_freq = base_freq
            self.time = time

            self._rate = self._calculate_jc_rate()

        def _jukes_cantor_correction(self, p):
            """
            Correct for multiple hits using the Jukes-Cantor model.

            Args:
                p (float): Observed proportion of differing nucleotides.

            Returns:
                float: Corrected distance D.
            """

            if p >= 3 / 4:
                raise ValueError(
                    f"The observed proportion of differing nucleotides ({p}) "
                    f"is too high for the Jukes-Cantor model. Try a different "
                    f"substitution model."
                )
            try:
                return -(3 / 4) * log(1 - (4 / 3) * p)
            except ValueError:
                raise ValueError(
                    f"The value of p ({p}) led to a math domain error in the "
                    f"Jukes-Cantor correction."
                )

        def _calculate_jc_rate(self, valid_bases={"A", "C", "G", "T"}):
            """
            Internal method to calculate mutation rate using Jukes-Cantor model.

            Returns:
                float: Mutation rate per site.
            """
            num_sequences = len(self.alignment)
            seq_length = self.alignment.get_alignment_length()
            total_mutations = 0
            for i in range(num_sequences):
                for j in range(i + 1, num_sequences):
                    mutations = sum(
                        1
                        for a, b in zip(self.alignment[i], self.alignment[j])
                        if a != b and a in valid_bases and b in valid_bases
                    )
                    total_mutations += mutations

            p = total_mutations / (
                seq_length * (num_sequences * (num_sequences - 1)) / 2
            )
            D = self._jukes_cantor_correction(p)

            if self.time:
                return D / (2 * self.time)
            else:
                return D / 2

        @property
        def rate(self):
            SNPulatorUtils.validate_positive_number(self._rate)
            return self._rate

    class HKY:
        def __init__(self, alignment, base_freq: dict, time: int) -> None:
            self.alignment = alignment
            self.base_freq = base_freq
            self.time = time

            self._rate = self._calculate_hky_rate()

        def _calculate_hky_Q(self):
            """
            Calculate the rate matrix (Q) for the HKY (Hasegawa-Kishino-Yano) model.

            Returns:
                np.ndarray: The rate matrix for the HKY model.
            """
            hky = msprime.HKY(
                SNPulatoRate._calculate_kappa(self.alignment),
                list(self.base_freq.values()),
            )
            return hky.transition_matrix
            # base_freq = self.base_freq

            # # Initialize the rate matrix with zeros
            # Q = np.zeros((4, 4))

            # # Order of bases
            # bases = ["A", "C", "G", "T"]

            # # Populate off-diagonal elements
            # for i, base1 in enumerate(bases):
            #     for j, base2 in enumerate(bases):
            #         if i != j:
            #             # Transitions (A <-> G, C <-> T)
            #             if {base1, base2} in [{"A", "G"}, {"C", "T"}]:
            #                 Q[i, j] = (
            #                     base_freq[base2] * self._calculate_kappa()
            #                 )
            #             # Transversions
            #             else:
            #                 Q[i, j] = base_freq[base2]

            # # Scale by mu (overall rate of substitution)
            # Q *= self._calculate_mu()

            # # Calculate diagonal elements so that each row sums to zero
            # for i in range(4):
            #     Q[i, i] = -np.sum(Q[i, :])

            # return Q

        def _calculate_hky_rate(self, valid_bases={"A", "C", "G", "T"}):
            """
            Calculate the mutation rate using the HKY (Hasegawa-Kishino-Yano) model.

            Returns:
                float: Mutation rate per site per time unit (if time is provided).
            """
            alignment = self.alignment

            # Get the rate matrix for the HKY model
            Q = self._calculate_hky_Q()

            # Initialize variables to hold the total mutations and the number
            # of valid sites
            total_mutations = 0
            num_sequences = len(alignment)
            seq_length = alignment.get_alignment_length()

            # Loop through each pair of sequences in the alignment
            for i in range(num_sequences):
                for j in range(i + 1, num_sequences):
                    # Loop through each position in the sequences
                    for pos in range(seq_length):
                        base1, base2 = alignment[i, pos], alignment[j, pos]
                        # Ignore gaps, missing data
                        if (
                            base1 not in valid_bases
                            or base2 not in valid_bases
                        ):
                            continue
                        # Map the bases to indices based on the order 'A', 'C',
                        # 'G', 'T'
                        index1 = ["A", "C", "G", "T"].index(base1)
                        index2 = ["A", "C", "G", "T"].index(base2)
                        # Get the rate from the rate matrix
                        rate = Q[index1, index2]
                        # Add the rate to the total mutations
                        total_mutations += rate

            # Calculate the mutation rate per site
            mutation_rate_per_site = total_mutations / (
                seq_length * (num_sequences * (num_sequences - 1) / 2)
            )

            # If time is provided, scale the mutation rate
            if self.time:
                mutation_rate = mutation_rate_per_site / self.time
                return mutation_rate
            else:
                return mutation_rate_per_site

        @property
        def rate(self):
            SNPulatorUtils.validate_positive_number(self._rate)
            return self._rate

    class F84:
        def __init__(self, alignment, base_freq: dict, time: int):
            self.alignment = alignment
            self.base_freq = base_freq
            self.time = time

        def _calculate_epsilon(self, valid_bases={"A", "C", "G", "T"}):
            """
            Estimate the epsilon parameter for the F84 model based on a given sequence alignment.
            Epsilon is an additional parameter to account for different GC content.

            Args:
                valid_bases (set): Set of valid bases to consider, defaults to {'A', 'C', 'G', 'T'}.

            Returns:
                float: Estimated epsilon value.
            """
            alignment = self.alignment
            gc_count = 0
            total_count = 0

            num_sequences = len(alignment)
            seq_length = alignment.get_alignment_length()

            # Count the number of G and C bases, and the total number of valid bases
            for i in range(num_sequences):
                for pos in range(seq_length):
                    base = alignment[i, pos]
                    if base in valid_bases:
                        total_count += 1
                        if base in ["G", "C"]:
                            gc_count += 1

            # Calculate the GC content
            gc_content = gc_count / total_count if total_count > 0 else 0

            # Estimate epsilon based on GC content
            epsilon = (gc_content / (1 - gc_content)) if gc_content < 1 else 0

            return epsilon

        def _calculate_f84_Q(self):
            """
            Calculate the rate matrix (Q) for the F84 (Felsenstein 1984) model.

            Returns:
                np.ndarray: The rate matrix for the F84 model.
            """
            f84 = msprime.F84(
                SNPulatoRate._calculate_kappa(self.alignment), self.base_freq
            )
            return f84.transition_matrix

        def _calculate_f84_mutation_rate(self):
            """
            Calculate the mutation rate using the F84 (Felsenstein 1984) model.

            Returns:
                float: Mutation rate per site per time unit (if time is provided).
            """
            alignment = self.alignment

            # Get the rate matrix for the F84 model
            Q = self._calculate_f84_Q()

            # Initialize variables to hold the total mutations and the number of valid sites
            total_mutations = 0
            num_sequences = len(alignment)
            seq_length = alignment.get_alignment_length()

            # Loop through each pair of sequences in the alignment
            for i in range(num_sequences):
                for j in range(i + 1, num_sequences):
                    # Loop through each position in the sequences
                    for pos in range(seq_length):
                        base1, base2 = alignment[i, pos], alignment[j, pos]
                        # Ignore gaps, missing data
                        if base1 in ["N", "?", "-", "."] or base2 in [
                            "N",
                            "?",
                            "-",
                            ".",
                        ]:
                            continue
                        # Map the bases to indices based on the order 'A', 'C', 'G', 'T'
                        index1 = ["A", "C", "G", "T"].index(base1)
                        index2 = ["A", "C", "G", "T"].index(base2)
                        # Get the rate from the rate matrix
                        rate = Q[index1, index2]
                        # Add the rate to the total mutations
                        total_mutations += rate

            # Calculate the mutation rate per site
            mutation_rate_per_site = total_mutations / (
                seq_length * (num_sequences * (num_sequences - 1) / 2)
            )

            # If time is provided, scale the mutation rate
            if self.time:
                mutation_rate = mutation_rate_per_site / self.time
                return mutation_rate
            else:
                return mutation_rate_per_site

        @property
        def rate(self):
            SNPulatorUtils.validate_positive_number(self._rate)
            return self._rate

    class Balanced(msprime.MatrixMutationModel):
        def __init__(self, *args):
            """
            Run an msprime simulation with a custom 10x10 rate matrix for nucleotides and IUPAC ambiguity codes.

            This class will attempt to give each integer-encoded IUPAC nucleotide equal probability of being simulated in an attempt to balance the nucleotides.

            Integer-encoded nucleotides correspond to 'A', 'C', 'G', 'T' (0-3), and 'R', 'Y', 'S', 'W', 'K', 'M' (4-9).
            """
            raise NotImplementedError("Balanced class is not implemented yet.")

            # Define the possible alleles: A, C, G, T and IUPAC ambiguity codes
            #  R, Y, S, W, K, M
            # int_iupac_dict = get_int_iupac_dict()
            alleles = ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M"]

            rate_matrix = np.full((10, 10), 0.1)
            np.fill_diagonal(rate_matrix, 0)

            # Normalize each row so that it sums to 1
            row_sums = rate_matrix.sum(axis=1, keepdims=True)
            rate_matrix = rate_matrix / row_sums

            # Define the root distribution: initial frequencies of each allele
            # For simplicity, let's assume they all have equal frequency
            # Example of a skewed root distribution
            root_distribution = [
                0.2,
                0.2,
                0.2,
                0.2,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
            ]
            root_distribution = [
                x / sum(root_distribution) for x in root_distribution
            ]  # Normalize to sum to 1

            super().__init__(alleles, root_distribution, rate_matrix)

        def asdict(self):
            # This version of asdict makes sure that we have sufficient
            #  parameters
            # to call the constructor and recreate the class. However, this
            # means
            # that subclasses *must* have an instance variable of the same name
            # as each parameter.
            # This is essential for Provenance round-tripping to work.
            return {
                key: getattr(self, key)
                for key in inspect.signature(self.__init__).parameters.keys()
                if hasattr(self, key)
            }

        @property
        def model(self):
            return self._model

    def calculate_rate(self, model="JC69"):
        """
        Calculate mutation rate based on the selected model.

        Args:
            model (str): Evolutionary model to use ('JC69' for Jukes-Cantor, 'GTR' for General Time Reversible, 'HKY' for Hasegawa-Kishino-Yano, and 'F84' for Felsenstein-84).

        Returns:
            float: Mutation rate per site.
        """
        args = [self.alignment, self.base_freq, self.time]
        if model.upper() == "JC69":
            m = self.JC69
        elif model.upper() == "GTR":
            m = self.GTR
        elif model.upper() == "HKY":
            m = self.HKY
        elif model.upper() == "F84":
            m = self.F84
        elif model.upper() == "BALANCED":
            m = self.Balanced
        else:
            raise ValueError(
                "Invalid model. Choose between 'JC69', 'GTR', 'HKY', and 'F84', 'Balanced'."
            )
        m_inst = m(*args)
        return m_inst.rate


# Configuration Class
class SNPulatorConfig:
    def __init__(
        self,
        sequence_length=1e4,
        mutation_rate=1e-8,
        recombination_rate=1e-7,
        mutation_model="jc69",
        recombination_map=None,
        include_pops=None,
        demes_graph=None,
        guidetree=False,
        msprime_model=None,
        record_migrations=True,
        root_divergence_time=1,
    ):
        """Configuration class for managing parameters of the SNPulator simulation.

        Attributes:
            sequence_length (float): Length of the sequence to be simulated. Defaults to 1e4.
            mutation_rate (float): Mutation rate per base per generation. Defaults to 1e-8.
            recombination_rate (float): Recombination rate per base per generation. Defaults to 1e-7.
            mutation_model (str or msprime mutation model instance): Name of the mutation model to be used or an msprime mutation model instance. Supported string options include: {'jc69', 'binary', and 'infinite_alleles'}. Supported model instances are: {GTR, HKY, F84, SLiMMutationModel}. Defaults to "jc69".
            recombination_map (str or None, optional): File path to the recombination map, if any. Defaults to None.
            include_pops (list or None, optional): List of population IDs to include in the simulation. Should correspond to populations in the demographic model. If set to None, will use all populations. Defaults to None.
            demes_graph (str, demes.Graph, or None, optional): Demes Graph object to set the demographic model for simulations, or a string indicating the file path to the demes YAML model file. If ``demes_graph``\, ``guidetree``\, or ``msprime_model`` are not provided, then you must use the SNPulator API to construct your demographic model with msprime.
            guidetree (bool): Whether to use GenotypeData.tree object to load the demographic model corresponding to the species tree. If ``demes_graph``\, ``guidetree``\, or ``msprime_model`` are not provided, then you must use the SNPulator API to construct your demographic model with msprime. Defaults to False.
            msprime_model (tskit.TreeSequence or None, optional): tskit.TreeSequence object to use to build the demographic model. If ``demes_graph``\, ``guidetree``\, or ``msprime_model`` are not provided, then you must use the SNPulator API to construct your demographic model with msprime.
            record_migrations (bool): Whether to record migration events when simulating the alignment(s). Defaults to True.
            root_divergence_time (int or None): Diverence time of ancestral population. If provided, the mutation rates are scaled to years. If time is 1, then the mutation rates are scaled to generations. Defaults to 1.

        """
        utils = SNPulatorUtils
        utils.validate_positive_number(sequence_length, "Sequence Length")
        utils.validate_positive_number(mutation_rate, "Mutation Rate")
        utils.validate_positive_number(
            root_divergence_time, "Root Divergence Time", min=1
        )
        utils.validate_num_range(mutation_rate, name="Mutation Rate")
        utils.validate_positive_number(
            recombination_rate, "Recombination Rate"
        )
        utils.validate_num_range(recombination_rate, name="Recombination Rate")
        utils.validate_mutation_model(mutation_model, "mutation_model")

        if recombination_map is not None:
            utils.validate_isfile(recombination_map, "Recombination Map")

        if include_pops is not None:
            utils.validate_type(include_pops, list, "include_pops")

        if demes_graph is not None:
            if not isinstance(demes_graph, str) and not isinstance(
                demes_graph, demes.Graph
            ):
                raise TypeError(
                    "Demes Graph must be of type demes.Graph or str"
                )
            elif isinstance(demes_graph, str):
                utils.validate_isfile(demes_graph, name="Demes Graph")

        if msprime_model is not None:
            utils.validate_type(
                msprime_model, tskit.TreeSequence, name="msprime model"
            )
            utils.validate_demographic_model(msprime_model)

        utils.validate_type(record_migrations, bool, "record_migrations")
        utils.validate_arglist(demes_graph, msprime_model, guidetree)

        self.sequence_length = sequence_length
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.mutation_model = mutation_model
        self.recombination_map = recombination_map
        self.include_pops = include_pops
        self.demes_graph = demes_graph
        self.guidetree = guidetree
        self.msprime_model = msprime_model
        self.record_migrations = record_migrations
        self.root_divergence_time = root_divergence_time

    # Method to update individual configuration parameters (Example)
    def update_mutation_rate(self, new_rate):
        """
        Update the mutation rate.

        Args:
            new_rate (float): The new mutation rate to set.

        """
        SNPulatorUtils.validate_positive_number(new_rate, "New Mutation Rate")
        self.mutation_rate = new_rate

    def update_recombination_rate(self, new_rate):
        """
        Update the recombination rate.

        Args:
            new_rate (float): The new recombination rate to set.

        """
        SNPulatorUtils.validate_positive_number(
            new_rate, "New Recombination Rate"
        )
        self.recombination_rate = new_rate

    def update_mutation_model(self, new_model):
        """
        Update the mutatation model.

        Args:
            new_model (msprime.MatrixMutationModel): Mutation model to set.
        """
        self.mutation_model = new_model

    def get_config(self):
        """
        Retrieve the current configuration as a dictionary.

        Returns:
            dict: A dictionary containing the current configuration parameters.

        """
        return {
            "sequence_length": self.sequence_length,
            "mutation_rate": self.mutation_rate,
            "recombination_rate": self.recombination_rate,
            "recombination_map": self.recombination_map,
            "include_pops": self.include_pops,
            "mutation_model": self.mutation_model,
            "demes_graph": self.demes_graph,
            "guidetree": self.guidetree,
            "msprime_model": self.msprime_model,
            "record_migrations": self.record_migrations,
            "root_divergence_time": self.root_divergence_time,
        }


class SNPulator:
    def __init__(self, genotype_data, config: SNPulatorConfig):
        """Initialize the SNPulator class.

        Args:
            genotype_data (GenotypeData): GenotypeData instance.

            config (SNPulatorConfig): SNPulatorConfig object containing necessary parameters.
        """
        self.genotype_data = genotype_data.copy()
        self.config = config.get_config()

        self.sequence_length = self.config["sequence_length"]
        self.recombination_rate = self.config["recombination_rate"]
        self.record_migrations = self.config["record_migrations"]
        self.time = self.config["root_divergence_time"]

        anc_kwargs = [
            "record_migrations",
            "recombination_rate",
            "sequence_length",
        ]

        mut_kwargs = ["mutation_rate", "mutation_model"]

        self.anc_config = {
            k: v for k, v in self.config.items() if k in anc_kwargs
        }

        self.mut_config = {}
        for k, v in self.config.items():
            if k in mut_kwargs:
                if k == "mutation_rate":
                    self.mut_config["rate"] = v
                elif k == "mutation_model":
                    self.mut_config["model"] = v

        self.recombination_map = self.config["include_pops"]
        self.demes_graph = self.config["demes_graph"]
        self.guidetree = self.config["guidetree"]
        self.msprime_model = self.config["msprime_model"]
        self.mutation_rate = self.config["mutation_rate"]
        self.mutation_model = self.config["mutation_model"]

        if self.demes_graph is not None:
            model = self.demes_graph
        elif self.guidetree:
            model = self.genotype_data.tree
        elif self.msprime_model is not None:
            model = self.msprime_model
        else:
            model = None

        # Initialize demography
        self.demography = self.initialize_demography(model)
        self.anc_config["demography"] = self.demography

        self.tree_sequence = None

    def initialize_demography(
        self, model: Optional[Union[str, demes.Graph, tskit.TreeSequence]]
    ):
        """Initialize the demography model based on provided parameters or custom model.

        Args:
            model (str, demes.Graph, or tskit.TreeSequence): If a string is provided, it must be a path to a demes YAML file. Otherwise, it should be a demes graph or a TreeSequence object.
        """
        SNPulatorUtils.validate_genotype_data_popmap(self.genotype_data)
        pops = self.genotype_data.popmap_inverse
        initial_size = {k: len(v) for k, v in pops.items()}
        if isinstance(model, str):
            demes_model = demes.load(model)
            model_obj = msprime.Demography.from_demes(demes_model)

        elif isinstance(model, tt.tree):
            tree = model.write()
            model_obj = msprime.Demography.from_species_tree(
                tree, initial_size
            )
        elif isinstance(model, tskit.TreeSequence):
            model_obj = msprime.Demography.from_tree_sequence(
                model, initial_size=initial_size
            )
        elif model is None:
            model_obj = msprime.Demography()
        else:
            raise TypeError(
                f"Invalid demographic model provided: {type(model)}"
            )

        return model_obj

    def sim_ancestry(self, sample_sizes, populations):
        """Simulate the tree sequence based on the specified parameters and demographic model."""
        self.sample_sets = self._create_sample_sets_dict(
            sample_sizes, populations
        )

        self._simulate_ancestry()
        return [
            f"seq{i}" for i in range(sum(sample_sizes))
        ], SNPulator.replicate_list_by_dict(populations, self.sample_sets)

    @staticmethod
    def replicate_list_by_dict(lst, rep_dict):
        """
        Replicates each element in a list based on a corresponding dictionary.

        Args:
            lst (list): The list containing elements to be replicated.
            rep_dict (dict): A dictionary where keys correspond to elements in `lst`
                            and values indicate the number of times each element should
                            be replicated.

        Returns:
            list: A new list where each element from `lst` is replicated according to `rep_dict`.
        """
        new_lst = []
        for item in lst:
            if item in rep_dict:
                new_lst.extend([item] * rep_dict[item])
            else:
                new_lst.append(
                    item
                )  # If the item is not in the dictionary, keep it as is
        return new_lst

    def _create_sample_sets_dict(self, sample_sizes, populations):
        """Create a dictionary of sample sets based on sample sizes and populations."""
        return dict(zip(populations, sample_sizes))

    def _simulate_ancestry(self):
        """Perform the ancestry simulation using msprime."""

        self.tree_sequence = msprime.sim_ancestry(
            samples=self.sample_sets,
            **self.anc_config,
        )

    def sim_mutations(self):
        """Overlay mutations on the simulated tree sequence."""
        snprate = SNPulatoRate(self.genotype_data, self.time)
        if "model" in self.mut_config:
            if self.mut_config["model"] in [
                msprime.HKY,
                msprime.F84,
                msprime.GTR,
                msprime.JC69,
                snprate.Balanced,
            ]:
                snprate = SNPulatoRate(self.genotype_data, self.time)
                self.base_freq = snprate._calculate_base_frequencies()
                m = self.mut_config["model"]
                if m in [msprime.HKY, msprime.F84]:
                    kappa = snprate._calculate_kappa(
                        self.genotype_data.alignment
                    )
                    m_init = m(
                        kappa=kappa,
                        equilibrium_frequencies=list(self.base_freq.values()),
                    )
                elif m == msprime.GTR:
                    gtr = snprate.GTR(
                        self.genotype_data.alignment, self.base_freq, self.time
                    )

                    r = gtr._calculate_relative_rates()

                    m_init = m(
                        relative_rates=r,
                        equilibrium_frequencies=list(self.base_freq.values()),
                    )
                elif m == msprime.JC69:
                    m_init = m()
                elif m == snprate.Balanced:
                    m_init = m()
            self.mut_config["model"] = m_init

            print(m_init)

        self.tree_sequence = msprime.sim_mutations(
            tree_sequence=self.tree_sequence,
            **self.mut_config,
        )

    def simulate_snp(self, sample_size, num_snps):
        replicates = msprime.sim_ancestry(
            samples=sample_size,
            population_size=0.5,
            num_replicates=100 * num_snps,
        )

        t_max = 0
        variants = np.empty((num_snps, sample_size), dtype="u1")
        total_branch_length = np.empty(num_snps)
        j = 0
        num_adaptive_updates = 0
        num_rejected_trees = 0
        for ts in replicates:
            tree = msprime.sim_mutations(ts, rate=10)
            tbl = tree.get_total_branch_length()
            if tbl > t_max:
                new_t_max = tbl
                new_variants = np.empty((num_snps, sample_size), dtype="u1")
                new_total_branch_length = np.empty(num_snps)
                keep = np.where(np.random.random(j) < t_max / new_t_max)[0]
                j = keep.shape[0]
                new_variants[:j] = variants[keep]
                new_total_branch_length[:j] = total_branch_length[keep]
                variants = new_variants
                total_branch_length = new_total_branch_length
                t_max = new_t_max
                num_adaptive_updates += 1
            else:
                if np.random.random() < tbl / t_max:
                    total_branch_length[j] = tbl
                    for variant in ts.variants():
                        variants[j] = variant.genotypes
                        break
                    else:
                        raise Exception("Must have at least one mutation")
                    j += 1
                    if j == num_snps:
                        break
                else:
                    num_rejected_trees += 1
        assert j == num_snps
        print("num adaptive updates: ", num_adaptive_updates)
        print("num rejected trees", num_rejected_trees)
        return variants

    def sim_ancestry_replicates(
        self, num_replicates, parallel=False, n_jobs=-1
    ):
        """Simulate multiple tree sequences, either sequentially or in parallel."""
        self._validate_parallel_settings(parallel, n_jobs)
        if parallel:
            return self._sim_ancestry_replicates_parallel(
                num_replicates, n_jobs
            )
        else:
            return self._sim_ancestry_replicates_sequential(num_replicates)

    def _validate_parallel_settings(self, parallel, n_jobs):
        """Validate settings for parallel execution."""
        if parallel and n_jobs == 1:
            raise ValueError(
                "Cannot use 1 CPU job with parallel execution. Set n_jobs to -1 or an integer > 1."
            )
        elif parallel and n_jobs < -1:
            raise ValueError(
                f"Invalid value for n_jobs: must be -1 or > 1, got {n_jobs}."
            )

    def _sim_ancestry_replicates_parallel(self, num_replicates, n_jobs):
        """Simulate multiple tree sequences in parallel and return segregating sites."""
        rng = np.random.RandomState(42)
        seeds = rng.randint(1, 2**31, size=(num_replicates, 2))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = executor.map(self._parallel_simulation_worker, seeds)
        return np.array(list(results))

    def _parallel_simulation_worker(self, seeds):
        """Worker function for parallel simulation."""
        ancestry_seed, mutation_seed = seeds
        ts = self._simulate_single_ancestry(ancestry_seed)
        mutated_ts = self._simulate_single_mutation(ts, mutation_seed)
        return mutated_ts.segregating_sites(span_normalise=False, mode="site")

    def _sim_ancestry_replicates_sequential(self, num_replicates):
        """Simulate multiple tree sequences sequentially and yield them one by one."""
        ancestry_reps = self._simulate_multiple_ancestries(num_replicates)
        for ts in ancestry_reps:
            mutated_ts = self._simulate_single_mutation(ts)
            yield mutated_ts

    def _simulate_multiple_ancestries(self, num_replicates):
        """Simulate multiple ancestries."""
        return msprime.sim_ancestry(
            samples=self.sample_sets,
            **self.anc_config,
            num_replicates=num_replicates,
        )

    def _simulate_single_ancestry(self, random_seed=None):
        """Simulate a single ancestry."""
        return msprime.sim_ancestry(
            samples=self.sample_sets,
            random_seed=random_seed,
            **self.anc_config,
        )

    def _simulate_single_mutation(self, ts, random_seed=None):
        """Simulate mutations on a single tree sequence."""
        return msprime.sim_mutations(
            ts,
            random_seed=random_seed,
            **self.mut_config,
        )

    class LDUtils:
        """Inner class for LD related utilities."""

        @staticmethod
        def ld_decay(x, c, r):
            """LD decay function."""
            return c * np.exp(-r * x)

        @staticmethod
        def fit_ld_decay(ld_matrix, bin_edges):
            """Fit a curve to the LD decay plot and estimate the recombination rate."""
            avg_ld = np.mean(ld_matrix, axis=1)
            popt, _ = curve_fit(SNPulator.LDUtils.ld_decay, bin_edges, avg_ld)
            return popt[1]

        @staticmethod
        def plot_ld_decay(ld_matrix, bin_edges, genotype_data, show=False):
            """Plot the LD decay curve."""
            avg_ld = np.mean(ld_matrix, axis=1)
            plt.figure(figsize=(10, 6))
            plt.scatter(bin_edges, avg_ld, label="Observed LD")
            plt.xlabel("Bin Edges")
            plt.ylabel("Average LD")
            plt.title("LD Decay Curve")
            plt.legend()

            if show:
                plt.show()

            plot_dir = os.path.join(
                f"{genotype_data.prefix}", "simulations", "plots"
            )
            Path(plot_dir).mkdir(exist_ok=True, parents=True)
            outfile = os.path.join(
                plot_dir,
                "ld_decay.png",
            )

            plt.savefig(outfile, facecolor="white")
            plt.close()

    def set_recombination_rate(
        self, strategy="auto", rate=None, ld_matrix=None, bin_edges=None
    ):
        """Set the recombination rate based on a strategy."""
        if strategy == "auto":
            if ld_matrix is None or bin_edges is None:
                raise ValueError(
                    "For auto strategy, ld_matrix and bin_edges must be "
                    "provided."
                )
            self.recombination_rate = self.LDUtils.fit_ld_decay(
                ld_matrix, bin_edges, self.genotype_data
            )
        elif strategy == "manual":
            if rate is None:
                raise ValueError("For manual strategy, rate must be provided.")
            self.recombination_rate = rate
        else:
            raise ValueError("Invalid strategy. Choose 'auto' or 'manual'.")

    def calculate_r_bins(self, min_value, max_value, num_bins):
        """Calculate logarithmically spaced bins for recombination rate."""
        return np.logspace(np.log10(min_value), np.log10(max_value), num_bins)

    @staticmethod
    def compute_ld_statistics(
        genotype_data,
        demes_graph,
        r_bins=[0, 1, 2],
    ):
        """
        Compute Linkage Disequilibrium (LD) statistics using moments library.

        Args:
            genotype_data (GenotypeData): GenotypeData instance.

            demes_graph (demes.graph): The demes graph object.

            r_bins (list): A list of rho bins to set the recombination rate. Defaults to p = 4Nr = [0, 1, 2].

        Returns:
            dict: Dictionary containing LD statistics. See momemnts API.

        Example:
            >>> snp = SNPulator(genotype_data, config)
            >>>
            >>> ld_stats = SNPulator.LDUtils.compute_ld_statistics(
            >>>     snp.genotype_data,
            >>>     snp.demes_graph,
            >>>     r_bins,
            >>> )
        """
        try:
            import moments.LD
        except (ImportError, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "The moments package must be installed to use the LD "
                "functionality."
            )

        if genotype_data.filetype != "vcf":
            raise TypeError(
                "Invalid Filetype: A VCF file must be supplied to GenotypeData "
                "to use the LD functionality."
            )

        if isinstance(r_bins, tuple):
            r_bins = SNPulator.calculate_logarithmic_r_bins(
                r_bins[0], r_bins[1], r_bins[2]
            )

            ld_stats = moments.LD.LDstats.from_demes(demes_graph)
            return ld_stats

    def set_recombination_rate(self, rate):
        """
        Set the recombination rate.

        Args:
            rate (float): The recombination rate to set.
        """
        self.recombination_rate = rate

    def calculate_r_bins(self, min_value, max_value, num_bins):
        """
        Calculate logarithmically spaced bins for recombination rate.

        Args:
            min_value (float): The minimum value for the bins.
            max_value (float): The maximum value for the bins.
            num_bins (int): The number of bins.

        Returns:
            np.ndarray: Logarithmically spaced bins.
        """
        return np.logspace(np.log10(min_value), np.log10(max_value), num_bins)

    def fit_ld_decay(self, ld_matrix, bin_edges):
        """
        Fit a curve to the LD decay plot and estimate the recombination rate.

        Args:
            ld_matrix (np.ndarray): The LD statistics matrix.
            bin_edges (np.ndarray): The edges of the bins used for LD calculation.

        Returns:
            float: Estimated recombination rate.
        """

        # Define the LD decay function
        def ld_decay(x, c, r):
            return c * np.exp(-r * x)

        # Calculate the average LD for each bin
        avg_ld = np.mean(ld_matrix, axis=1)

        # Fit the LD decay curve
        popt, _ = curve_fit(ld_decay, bin_edges, avg_ld)

        # Extract the recombination rate from the fitted parameters
        estimated_r = popt[1]

        return estimated_r

    def auto_set_recombination_rate(self, ld_matrix, bin_edges):
        """
        Automatically set the recombination rate by fitting an LD decay curve.

        Args:
            ld_matrix (np.ndarray): The LD statistics matrix.
            bin_edges (np.ndarray): The edges of the bins used for LD calculation.
        """
        estimated_r = self.fit_ld_decay(ld_matrix, bin_edges)
        self.set_recombination_rate(estimated_r)

        # Plot for visualization
        self.plot_ld_decay(ld_matrix, bin_edges)

    def plot_ld_decay(self, ld_matrix, bin_edges):
        """
        Plot the LD decay curve.

        Args:
            ld_matrix (np.ndarray): The LD statistics matrix.
            bin_edges (np.ndarray): The edges of the bins used for LD calculation.
        """
        # Calculate the average LD for each bin
        avg_ld = np.mean(ld_matrix, axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_edges, avg_ld, label="Observed LD")
        plt.xlabel("Bin Edges")
        plt.ylabel("Average LD")
        plt.title("LD Decay Curve")
        plt.legend()
        plt.show()

    def _add_monomorphic_sites(self, tree_sequence, ancestral_states):
        tables = tree_sequence.dump_tables()
        sequence_length = int(tree_sequence.sequence_length)

        # Calculate the positions that are already occupied
        existing_positions = set(
            site.position for site in tree_sequence.sites()
        )

        # Calculate the positions that are not occupied
        all_positions = set(range(sequence_length))
        empty_positions = all_positions - existing_positions

        # Get the original base frequencies
        base_frequencies = self.base_freq

        # Normalize the frequencies so they sum to 1
        total_frequency = sum(base_frequencies.values())
        normalized_frequencies = {
            k: v / total_frequency for k, v in base_frequencies.items()
        }

        # Create a list of ancestral states based on their frequencies
        ancestral_states_pool = []
        for state, freq in normalized_frequencies.items():
            num_occurrences = int(freq * len(empty_positions))
            ancestral_states_pool.extend([state] * num_occurrences)

        # If there are still empty positions left, fill them randomly
        remaining_positions = len(empty_positions) - len(ancestral_states_pool)
        if remaining_positions > 0:
            extra_states = np.random.choice(
                ancestral_states, remaining_positions
            )
            ancestral_states_pool.extend(extra_states)

        # Shuffle the ancestral states to randomize their positions
        np.random.shuffle(ancestral_states_pool)

        # Add the monomorphic sites
        for pos, state in zip(empty_positions, ancestral_states_pool):
            tables.sites.add_row(position=pos, ancestral_state=state)

        tables.sort()
        return tables.tree_sequence()

    @staticmethod
    def to_iupac(nucleotide_tuple):
        iupac_dict = {
            frozenset(["A", "G"]): "R",
            frozenset(["C", "T"]): "Y",
            frozenset(["G", "C"]): "S",
            frozenset(["A", "T"]): "W",
            frozenset(["G", "T"]): "K",
            frozenset(["A", "C"]): "M",
            frozenset(["A", "A"]): "A",
            frozenset(["C", "C"]): "C",
            frozenset(["G", "G"]): "G",
            frozenset(["T", "T"]): "T",
        }
        return iupac_dict.get(frozenset(nucleotide_tuple), "N")

    def _convert_to_iupac_2d(self, full_genotypes):
        n_samples, n_sites = full_genotypes.shape
        iupac_2d_list = []

        for i in range(n_samples):
            iupac_row = []
            for j in range(n_sites):
                nucleotide_tuple_list = full_genotypes[i, j]
                # Take the first tuple from the list (assuming all tuples in the list are the same)
                first_tuple = (
                    nucleotide_tuple_list[0]
                    if nucleotide_tuple_list
                    else ("N", "N")
                )
                iupac_code = SNPulator.to_iupac(first_tuple)
                iupac_row.append(iupac_code)
            iupac_2d_list.append(iupac_row)

        return iupac_2d_list

    @property
    def genotypes(self):
        tree_sequence = self.tree_sequence

        # Initialize an empty list to store the full genotypes
        full_genotypes = []

        # Iterate through the variants in the tree sequence
        for variant in tree_sequence.variants():
            decoded_gt = np.array(
                [variant.alleles[g] for g in variant.genotypes]
            )
            decoded_gt = decoded_gt.reshape(-1, 2)

            # Convert the decoded genotypes to IUPAC codes
            iupac_row = [self.to_iupac((a1, a2)) for a1, a2 in decoded_gt]
            full_genotypes.append(iupac_row)

        # Transpose the list of lists to match the original shape
        full_genotypes = list(map(list, zip(*full_genotypes)))

        return full_genotypes
