import os
import sys
import warnings

from typing import Optional, Union, List, Dict, Tuple, Any, Callable

# Make sure python version is >= 3.6
if sys.version_info < (3, 6):
    raise ImportError("Python < 3.6 is not supported!")

import numpy as np
import pandas as pd
import toytree as tt

from read_input.popmap_file import ReadPopmap
from utils import sequence_tools


class GenotypeData:
    """Read genotype and tree data and encode genotypes.

    Reads in a PHYLIP or STRUCTURE-formatted input file and converts the genotypes to 012 or one-hot encodings.

    Args:
            filename (str or None): Path to input file containing genotypes. Defaults to None.

            filetype (str or None): Type of input genotype file. Possible ``filetype`` values include: "phylip", "structure1row", or "structure2row". VCF compatibility may be added in the future, but is not currently supported. Defaults to None.

            popmapfile (str or None): Path to population map file. If ``popmapfile`` is supplied and ``filetype`` is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column. Defaults to None.

            guidetree (str or None): Path to input treefile. Defaults to None.

            qmatrix_iqtree (str or None): Path to iqtree output file containing Q rate matrix. Defaults to None.

            qmatrix (str or None): Path to file containing only Q rate matrix, and not the full iqtree file. Defaults to None.

            verbose (bool, optional): Verbosity level. Defaults to True.

    Attributes:
            samples (List[str]): List containing sample IDs of shape (n_samples,).

            snps (List[List[str]]): 2D list of shape (n_samples, n_sites) containing genotypes.

            pops (List[str]): List of population IDs of shape (n_samples,).

            onehot (List[List[List[float]]]): One-hot encoded genotypes as a 3D list of shape (n_samples, n_sites, 4). The inner-most list represents the four nucleotide bases in the order of "A", "T", "G", "C". If position 0 contains a 1.0, then the site is an "A". If position 1 contains a 1.0, then the site is a "T"...etc. Two values of 0.5 indicates a heterozygote. Missing data is encoded as four values of 0.0.

            guidetree (toytree object): Input guide tree as a toytree object.

            num_snps (int): Number of SNPs (features) present in the dataset.

            num_inds: (int): Number of individuals (samples) present in the dataset.

    Properties:
            snpcount (int): Number of SNPs (features) in the dataset.

            indcount (int): Number of individuals (samples) in the dataset.

            populations (List[str]): List of population IDs of shape (n_samples,).

            individuals (List[str]): List of sample IDs of shape (n_samples,).

            genotypes_list (List[List[str]]): List of 012-encoded genotypes of shape (n_samples, n_sites).

            genotypes_nparray (numpy.ndarray): 012-encoded genotypes of shape (n_samples, n_sites).

            genotypes_df (pandas.DataFrame): 012-encoded genotypes of shape (n_samples, n_sites). Missing values are encoded as -9.

            genotypes_onehot (numpy.ndarray of shape (n_samples, n_SNPs, 4)): One-hot encoded numpy array. The inner-most array consists of one-hot encoded values for the four nucleotides in the order of "A", "T", "G", "C". Values of 0.5 indicate heterozygotes, and missing values contain 0.0 for all four nucleotides.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        filetype: Optional[str] = None,
        popmapfile: Optional[str] = None,
        guidetree: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.filename = filename
        self.filetype = filetype
        self.popmapfile = popmapfile
        self.guidetree = guidetree
        self.qmatrix_iqtree = qmatrix_iqtree
        self.qmatrix = qmatrix
        self.verbose = verbose

        self.snpsdict: Dict[str, List[Union[str, int]]] = dict()
        self.samples: List[str] = list()
        self.snps: List[List[int]] = list()
        self.pops: List[Union[str, int]] = list()
        self.onehot: Union[np.ndarray, List[List[List[float]]]] = list()
        self.ref = list()
        self.alt = list()
        self.num_snps: int = 0
        self.num_inds: int = 0

        if self.qmatrix_iqtree is not None and self.qmatrix is not None:
            raise TypeError("qmatrix_iqtree and qmatrix cannot both be defined")

        if self.filetype is not None:
            self.parse_filetype(filetype, popmapfile)

        if self.popmapfile is not None:
            self.read_popmap(popmapfile)

        if self.guidetree is not None:
            self.tree = self.read_tree(self.guidetree)
        elif self.guidetree is None:
            self.tree = None

        if self.qmatrix_iqtree is not None:
            self.q = self.q_from_iqtree(self.qmatrix_iqtree)
        elif self.qmatrix_iqtree is None and self.qmatrix is not None:
            self.q = self.q_from_file(self.qmatrix)
        elif self.qmatrix is None and self.qmatrix_iqtree is None:
            self.q = None

    def parse_filetype(
        self, filetype: Optional[str] = None, popmapfile: Optional[str] = None
    ) -> None:
        """Check the filetype and call the appropriate function to read the file format.

        Args:
            filetype (str or None): Filetype. Supported values include: "phylip", "structure1row", "structure2row", "structure1rowPopID", and "structure2rowPopID". Defaults to None.

            popmapfile (str or None): Path to population map file. Defaults to None.

        Raises:
            OSError: No filetype specified.
            OSError: Filetype not supported.
        """
        if filetype is None:
            raise OSError("No filetype specified.\n")
        else:
            if filetype == "phylip":
                self.filetype = filetype
                self.read_phylip()
            elif filetype.lower().startswith("structure1row"):
                if popmapfile is not None and filetype.lower().endswith("row"):
                    self.filetype = "structure1row"
                    self.read_structure(onerow=True, popids=False)

                elif popmapfile is None and filetype.lower().endswith("popid"):
                    self.filetype = "structure1rowPopID"
                    self.read_structure(onerow=True, popids=True)

                elif popmapfile is not None and filetype.lower().endswith(
                    "popid"
                ):
                    print(
                        "WARNING: popmapfile was not None but provided "
                        "filetype was structure1rowPopID. Using populations "
                        "from 2nd column in STRUCTURE file."
                    )
                    self.filetype = "structure1rowPopID"
                    self.read_structure(onerow=True, popids=True)

                elif popmapfile is None and filetype.lower().endswith("row"):
                    raise ValueError(
                        "If popmap file is not provided, filetype must be "
                        "structure1rowPopID and the 2nd STRUCTURE file column "
                        "should contain population IDs"
                    )

                else:
                    raise ValueError(
                        f"Unsupported filetype provided: {filetype}"
                    )

            elif filetype.lower().startswith("structure2row"):
                if popmapfile is not None and filetype.lower().endswith("row"):
                    self.filetype = "structure2row"
                    self.read_structure(onerow=False, popids=False)

                elif popmapfile is None and filetype.lower().endswith("popid"):
                    self.filetype = "structure2rowPopID"
                    self.read_structure(onerow=False, popids=True)

                elif popmapfile is not None and filetype.lower().endswith(
                    "popid"
                ):
                    print(
                        "WARNING: popmapfile was not None but provided "
                        "filetype was structure2rowPopID. Using populations "
                        "from 2nd column in STRUCTURE file."
                    )
                    self.filetype = "structure2rowPopID"
                    self.read_structure(onerow=False, popids=True)

                elif popmapfile is None and filetype.lower().endswith("row"):
                    raise ValueError(
                        "If popmap file is not provided, filetype must be "
                        "structure2rowPopID and the 2nd STRUCTURE file column "
                        "should contain population IDs"
                    )

                else:
                    raise OSError(f"Unsupported filetype provided: {filetype}")

            else:
                raise OSError(f"Unsupported filetype provided: {filetype}\n")

    def check_filetype(self, filetype: str) -> None:
        """Validate that the filetype is correct.

        Args:
            filetype (str or None): Filetype to use.

        Raises:
            TypeError: Filetype does not match the validation.
        """
        if self.filetype is None:
            self.filetype = filetype
        elif self.filetype == filetype:
            pass
        else:
            raise TypeError(
                "GenotypeData read_XX() call does not match filetype!\n"
            )

    def read_tree(self, treefile: str) -> tt.tree:
        """Read Newick-style phylogenetic tree into toytree object.

        Format should be of type 0 (see toytree documentation).

        Args:
            treefile (str): Path to Newick-style tree file.

        Returns:
            toytree.tree object: Input tree as toytree object.
        """
        if not os.path.isfile(treefile):
            raise FileNotFoundError(f"File {treefile} not found!")

        assert os.access(treefile, os.R_OK), f"File {treefile} isn't readable"

        return tt.tree(treefile, tree_format=0)

    def q_from_file(self, fname: str, label: bool = True) -> pd.DataFrame:
        """Read Q matrix from file on disk.

        Args:
            fname (str): Path to Q matrix input file.

            label (bool): True if nucleotide label order is present, otherwise False.

        Returns:
            pandas.DataFrame: Q-matrix as pandas DataFrame object.
        """
        q = self.blank_q_matrix()

        if not label:
            print(
                "Warning: Assuming the following nucleotide order: A, C, G, T"
            )

        with open(fname, "r") as fin:
            header = True
            qlines = list()
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if header:
                    if label:
                        order = line.split()
                        header = False
                    else:
                        order = ["A", "C", "G", "T"]
                    continue
                else:
                    qlines.append(line.split())
        fin.close()

        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])
        qdf = pd.DataFrame(q)
        return qdf.T

    def q_from_iqtree(self, iqfile: str) -> pd.DataFrame:
        """Read in Q-matrix from \*.iqtree file.

        The \*.iqtree file is one of the IQ-TREE output files and contains the standard output of the IQ-TREE run.

        Args:
            iqfile (str): Path to \*.iqtree file.

        Returns:
            pandas.DataFrame: Q-matrix as pandas DataFrame.

        Raises:
            FileNotFoundError: If iqtree file could not be found.
            IOError: If iqtree file could not be read from.
        """
        q = self.blank_q_matrix()
        qlines = list()
        try:
            with open(iqfile, "r") as fin:
                foundLine = False
                matlinecount = 0
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    if "Rate matrix Q" in line:
                        foundLine = True
                        continue
                    if foundLine:
                        matlinecount += 1
                        if matlinecount > 4:
                            break
                        stuff = line.split()
                        qlines.append(stuff)
                    else:
                        continue
        except (IOError, FileNotFoundError):
            sys.exit(f"Could not open iqtree file {iqfile}")

        # Population q matrix with values from iqtree file
        order = [l[0] for l in qlines]
        for l in qlines:
            for index in range(0, 4):
                q[l[0]][order[index]] = float(l[index + 1])

        qdf = pd.DataFrame(q)
        return qdf.T

    def blank_q_matrix(
        self, default: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        q: Dict[str, Dict[str, float]] = dict()
        for nuc1 in ["A", "C", "G", "T"]:
            q[nuc1] = dict()
            for nuc2 in ["A", "C", "G", "T"]:
                q[nuc1][nuc2] = default
        return q

    def read_structure(self, onerow: bool = False, popids: bool = True) -> None:
        """Read a structure file with two rows per individual.

        Args:
            onerow (bool, optional): True if file is in one-row format. False if two-row format. Defaults to False.

            popids (bool, optional): True if population IDs are present as 2nd column in structure file, otherwise False. Defaults to True.

        Raises:
            ValueError: Sample names do not match for two-row format.
            ValueError: Population IDs do not match for two-row format.
            AssertionError: All sequences must be the same length.
        """
        if self.verbose:
            print(f"\nReading structure file {self.filename}...")

        snp_data = list()
        with open(self.filename, "r") as fin:
            if not onerow:
                firstline = None
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    if not firstline:
                        firstline = line.split()
                        continue
                    else:
                        secondline = line.split()
                        if firstline[0] != secondline[0]:
                            raise ValueError(
                                f"Two rows per individual was "
                                f"specified but sample names do not match: "
                                f"{firstline[0]} and {secondline[0]}\n"
                            )

                        ind = firstline[0]
                        pop = None
                        if popids:
                            if firstline[1] != secondline[1]:
                                raise ValueError(
                                    f"Two rows per individual was "
                                    f"specified but population IDs do not "
                                    f"match {firstline[1]} {secondline[1]}\n"
                                )
                            pop = firstline[1]
                            self.pops.append(pop)
                            firstline = firstline[2:]
                            secondline = secondline[2:]
                        else:
                            firstline = firstline[1:]
                            secondline = secondline[1:]
                        self.samples.append(ind)
                        genotypes = merge_alleles(firstline, secondline)
                        snp_data.append(genotypes)
                        self.snpsdict[ind] = genotypes
                        firstline = None
            else:  # If onerow:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    firstline = line.split()
                    ind = firstline[0]
                    pop = None
                    if popids:
                        pop = firstline[1]
                        self.pops.append(pop)
                        firstline = firstline[2:]
                    else:
                        firstline = firstline[1:]
                    self.samples.append(ind)
                    genotypes = merge_alleles(firstline, second=None)
                    snp_data.append(genotypes)
                    self.snpsdict[ind] = genotypes
                    firstline = None

        if self.verbose:
            print("Done!")
            print("\nConverting genotypes to one-hot encoding...")

        # Convert snp_data to onehot encoding format
        self.convert_onehot(snp_data)

        if self.verbose:
            print("Done!")
            print("\nConverting genotypes to 012 format...")

        # Convert snp_data to 012 format
        self.convert_012(snp_data, vcf=True)

        if self.verbose:
            print("Done!")

        # Get number of samples and snps
        self.num_snps = len(self.snps[0])
        self.num_inds = len(self.samples)

        if self.verbose:
            print(
                f"\nFound {self.num_snps} SNPs and {self.num_inds} "
                f"individuals...\n"
            )

        # Make sure all sequences are the same length.
        for item in self.snps:
            try:
                assert len(item) == self.num_snps
            except AssertionError:
                sys.exit(
                    "There are sequences of different lengths in the "
                    "structure file\n"
                )

    def read_phylip(self) -> None:
        """Populates GenotypeData object by parsing Phylip.

        Raises:
            ValueError: All sequences must be the same length as specified in the header line.

            ValueError: Number of individuals differs from header line.
        """
        if self.verbose:
            print(f"\nReading phylip file {self.filename}...")

        self.check_filetype("phylip")
        snp_data = list()
        with open(self.filename, "r") as fin:
            num_inds = 0
            num_snps = 0
            first = True
            for line in fin:
                line = line.strip()
                if not line:  # If blank line.
                    continue
                if first:
                    first = False
                    header = line.split()
                    num_inds = int(header[0])
                    num_snps = int(header[1])
                    continue
                cols = line.split()
                inds = cols[0]
                seqs = cols[1]
                snps = [snp for snp in seqs]  # Split each site.

                # Error handling if incorrect sequence length
                if len(snps) != num_snps:
                    raise ValueError(
                        "All sequences must be the same length; "
                        "at least one sequence differs from the header line\n"
                    )

                self.snpsdict[inds] = snps
                snp_data.append(snps)

                self.samples.append(inds)

        if self.verbose:
            print("Done!")
            print("\nConverting genotypes to one-hot encoding...")

        # Convert snp_data to onehot format.
        self.convert_onehot(snp_data)

        if self.verbose:
            print("Done!")

            print("\nConverting genotypes to 012 encoding...")

        # Convert snp_data to 012 format
        self.convert_012(snp_data)

        if self.verbose:
            print("Done!")

        self.num_snps = num_snps
        self.num_inds = num_inds

        # Error handling if incorrect number of individuals in header.
        if len(self.samples) != num_inds:
            raise ValueError(
                "Incorrect number of individuals listed in header\n"
            )

    def read_phylip_tree_imputation(self, aln: str) -> Dict[str, List[str]]:
        """Function to read an alignment file.

        Args:
            aln (str): Path to alignment file.

        Returns:
            Dict[str, List[str]]: Dictionary with keys=sampleIDs and values=lists of sequences divided by site (i.e., all sites for one sample across all columns).

        Raises:
            TypeError: Alignment file not specified.
            IOError: Alignment file could not be read from.
            FileNotFoundError: Alignment file not found.
        """
        if aln is None:
            raise TypeError(
                "alignment file must be specified if using PHYLIP input "
                "format, but got NoneType"
            )

        elif os.path.exists(aln):
            with open(aln, "r") as fh:
                try:
                    num = 0
                    ret = dict()
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        num += 1
                        if num == 1:
                            continue
                        arr = line.split()
                        ret[arr[0]] = list(arr[1])
                    return ret

                except IOError:
                    print(f"Could not read file {aln}")
                    sys.exit(1)
                finally:
                    fh.close()
        else:
            raise FileNotFoundError(f"File {aln} not found!")

    def convert_012(
        self,
        snps: List[List[str]],
        vcf: bool = False,
        impute_mode: bool = False,
    ) -> List[List[int]]:
        """Encode IUPAC nucleotides as 0 (reference), 1 (heterogygous), and 2 (alternate) alleles.

        Args:
            snps (List[List[str]]): 2D list of genotypes of shape (n_samples, n_sites).

            vcf (bool, optional): Whether or not VCF file input is provided. Not yet supported. Defaults to False.

            impute_mode (bool, optional): Whether or not convert_012() is called in impute mode. If True, then returns the 012-encoded genotypes and does not set the ``self.snps`` attribute. If False, it does the opposite. Defaults to False.

        Returns:
            List[List[int]], optional: 012-encoded genotypes as a 2D list of shape (n_samples, n_sites). Only returns value if ``impute_mode`` is True.

            List[int], optional: List of integers indicating bi-allelic site indexes.

            int, optional: Number of remaining valid sites.
        """
        warnings.formatwarning = self._format_warning

        skip = 0
        new_snps = list()

        if impute_mode:
            imp_snps = list()

        for i in range(0, len(snps)):
            new_snps.append([])

        # TODO: valid_sites is now deprecated.
        valid_sites = np.ones(len(snps[0]))

        for j in range(0, len(snps[0])):
            loc = list()
            for i in range(0, len(snps)):
                if vcf:
                    loc.append(snps[i][j])
                else:
                    loc.append(snps[i][j].upper())

            num_alleles = sequence_tools.count_alleles(loc, vcf=vcf)
            if num_alleles != 2:

                # If monomorphic
                if num_alleles < 2:
                    warnings.warn(
                        f"Monomorphic site detected at SNP column {j+1}.\n"
                    )
                    ref = sequence_tools.get_major_allele(loc, vcf=vcf)
                    ref = str(ref[0])
                    alt = None

                    if vcf:
                        for i in range(0, len(snps)):
                            gen = snps[i][j].split("/")
                            if gen[0] in ["-", "-9", "N"] or gen[1] in [
                                "-",
                                "-9",
                                "N",
                            ]:
                                new_snps[i].append(-9)

                            elif gen[0] == gen[1] and gen[0] == ref:
                                new_snps[i].append(0)

                            else:
                                new_snps[i].append(1)
                    else:
                        for i in range(0, len(snps)):
                            if loc[i] in ["-", "-9", "N"]:
                                new_snps[i].append(-9)

                            elif loc[i] == ref:
                                new_snps[i].append(0)

                            else:
                                new_snps[i].append(1)

                # If >2 alleles
                elif num_alleles > 2:
                    warnings.warn(
                        f" SNP column {j+1} had >2 alleles and was forced to "
                        f"be bi-allelic. If that is not what you want, please "
                        f"fix or remove the column and re-run.\n"
                    )
                    all_alleles = sequence_tools.get_major_allele(loc, vcf=vcf)
                    all_alleles = [str(x[0]) for x in all_alleles]
                    ref = all_alleles.pop(0)
                    alt = all_alleles.pop(0)
                    others = all_alleles

                    if vcf:
                        for i in range(0, len(snps)):
                            gen = snps[i][j].split("/")
                            if gen[0] in ["-", "-9", "N"] or gen[1] in [
                                "-",
                                "-9",
                                "N",
                            ]:
                                new_snps[i].append(-9)

                            elif gen[0] == gen[1] and gen[0] == ref:
                                new_snps[i].append(0)

                            elif gen[0] == gen[1] and gen[0] == alt:
                                new_snps[i].append(2)

                            # Force biallelic
                            elif gen[0] == gen[1] and gen[0] in others:
                                new_snps[i].append(2)

                            else:
                                new_snps[i].append(1)
                    else:
                        for i in range(0, len(snps)):
                            if loc[i] in ["-", "-9", "N"]:
                                new_snps[i].append(-9)

                            elif loc[i] == ref:
                                new_snps[i].append(0)

                            elif loc[i] == alt:
                                new_snps[i].append(2)

                            # Force biallelic
                            elif loc[i] in others:
                                new_snps[i].append(2)

                            else:
                                new_snps[i].append(1)
                    # skip += 1
                    # valid_sites[j] = np.nan
                    # continue
            else:
                ref, alt = sequence_tools.get_major_allele(loc, vcf=vcf)
                ref = str(ref)
                alt = str(alt)

                if vcf:
                    for i in range(0, len(snps)):
                        gen = snps[i][j].split("/")
                        if gen[0] in ["-", "-9", "N"] or gen[1] in [
                            "-",
                            "-9",
                            "N",
                        ]:
                            new_snps[i].append(-9)

                        elif gen[0] == gen[1] and gen[0] == ref:
                            new_snps[i].append(0)

                        elif gen[0] == gen[1] and gen[0] == alt:
                            new_snps[i].append(2)

                        else:
                            new_snps[i].append(1)
                else:
                    for i in range(0, len(snps)):
                        if loc[i] in ["-", "-9", "N"]:
                            new_snps[i].append(-9)

                        elif loc[i] == ref:
                            new_snps[i].append(0)

                        elif loc[i] == alt:
                            new_snps[i].append(2)

                        else:
                            new_snps[i].append(1)

            # Set the ref and alt alleles for each column
            self.ref.append(ref)
            self.alt.append(alt)

        # TODO: skip and impute_mode are now deprecated.
        if skip > 0:
            if impute_mode:
                print(
                    f"\nWarning: Skipping {skip} non-biallelic sites following "
                    "imputation\n"
                )
            else:
                print(f"\nWarning: Skipping {skip} non-biallelic sites\n")

        for s in new_snps:
            if impute_mode:
                imp_snps.append(s)
            else:
                self.snps.append(s)

        if impute_mode:
            return (
                imp_snps,
                valid_sites,
                np.count_nonzero(~np.isnan(valid_sites)),
            )

    def _format_warning(
        self, message, category, filename, lineno, file=None, line=None
    ):
        """For setting the format of warnings.warn warnings.

        Set ``warnings.formatwarnings = self._format_warning`` to use it.

        Args:
            message (str): Warning message to print.
            category (str): Type of warning.
            filename (str): Name of python file where the warning was raised.
            lineno (str): Line number where warning occurred.
            file (None): Not used here.
            line (None): Not used here.

        Returns:
            str: Full warning message.
        """
        return f"{filename}:{lineno}: {category.__name__}:{message}"

    def convert_onehot(
        self,
        snp_data: Union[np.ndarray, List[List[int]]],
        encodings_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """Convert input data to one-hot format.

        Args:
            snp_data (numpy.ndarray of shape (n_samples, n_SNPs) or List[List[int]]): Input 012-encoded data.

            encodings_dict (Dict[str, int] or None): Encodings to convert structure to phylip format.

        Returns:
            numpy.ndarray: One-hot encoded data.
        """

        if self.filetype == "phylip" and encodings_dict is None:
            onehot_dict = {
                "A": [1.0, 0.0, 0.0, 0.0],
                "T": [0.0, 1.0, 0.0, 0.0],
                "G": [0.0, 0.0, 1.0, 0.0],
                "C": [0.0, 0.0, 0.0, 1.0],
                "N": [0.0, 0.0, 0.0, 0.0],
                "W": [0.5, 0.5, 0.0, 0.0],
                "R": [0.5, 0.0, 0.5, 0.0],
                "M": [0.5, 0.0, 0.0, 0.5],
                "K": [0.0, 0.5, 0.5, 0.0],
                "Y": [0.0, 0.5, 0.0, 0.5],
                "S": [0.0, 0.0, 0.5, 0.5],
                "-": [0.0, 0.0, 0.0, 0.0],
            }

        elif (
            self.filetype.startswith("structure1row")
            or self.filetype.startswith("structure2row")
            and encodings_dict is None
        ):
            onehot_dict = {
                "1/1": [1.0, 0.0, 0.0, 0.0],
                "2/2": [0.0, 1.0, 0.0, 0.0],
                "3/3": [0.0, 0.0, 1.0, 0.0],
                "4/4": [0.0, 0.0, 0.0, 1.0],
                "-9/-9": [0.0, 0.0, 0.0, 0.0],
                "1/2": [0.5, 0.5, 0.0, 0.0],
                "2/1": [0.5, 0.5, 0.0, 0.0],
                "1/3": [0.5, 0.0, 0.5, 0.0],
                "3/1": [0.5, 0.0, 0.5, 0.0],
                "1/4": [0.5, 0.0, 0.0, 0.5],
                "4/1": [0.5, 0.0, 0.0, 0.5],
                "2/3": [0.0, 0.5, 0.5, 0.0],
                "3/2": [0.0, 0.5, 0.5, 0.0],
                "2/4": [0.0, 0.5, 0.0, 0.5],
                "4/2": [0.0, 0.5, 0.0, 0.5],
                "3/4": [0.0, 0.0, 0.5, 0.5],
                "4/3": [0.0, 0.0, 0.5, 0.5],
            }

        else:
            if isinstance(snp_data, np.ndarray):
                snp_data = snp_data.tolist()

            onehot_dict = encodings_dict

        onehot_outer_list = list()

        if encodings_dict is None:
            for i in range(len(self.samples)):
                onehot_list = list()
                for j in range(len(snp_data[0])):
                    onehot_list.append(onehot_dict[snp_data[i][j]])
                onehot_outer_list.append(onehot_list)

            self.onehot = np.array(onehot_outer_list)

        else:
            for i in range(len(snp_data)):
                onehot_list = list()
                for j in range(len(snp_data[0])):
                    onehot_list.append(onehot_dict[snp_data[i][j]])
                onehot_outer_list.append(onehot_list)

            return np.array(onehot_outer_list)

    def read_popmap(self, popmapfile: Optional[str]) -> None:
        """Read population map from file.

        Args:
            popmapfile (str): Path to population map file.

        Raises:
            ValueError: No samples were in the input file.
            ValueError: Samples missing from the popmap file.
            ValueError: Lengths of popmap file and samples differ.
        """
        self.popmapfile = popmapfile
        # Join popmap file with main object.
        if len(self.samples) < 1:
            raise ValueError("No samples in GenotypeData\n")

        # Instantiate popmap object
        my_popmap = ReadPopmap(popmapfile)

        popmapOK = my_popmap.validate_popmap(self.samples)

        if not popmapOK:
            raise ValueError(
                f"Not all samples are present in supplied popmap "
                f"file: {my_popmap.filename}\n"
            )

        if len(my_popmap) != len(self.samples):
            raise ValueError(
                f"The number of individuals in the popmap file "
                f"({len(my_popmap)}) differs from the number of samples "
                f"({len(self.samples)})\n"
            )

        for sample in self.samples:
            if sample in my_popmap:
                self.pops.append(my_popmap[sample])

    def decode_imputed(self, X, write_output=True, prefix="output"):
        """Decode 012-encoded imputed data to STRUCTURE or PHYLIP format.

        Args:
            X (pandas.DataFrame, numpy.ndarray, or List[List[int]]): 012-encoded imputed data to decode.

            write_output (bool, optional): If True, saves output to file on disk. Otherwise just makes a GenotypeData attribute. Defaults to True.

            prefix (str, optional): Prefix to append to output file. Defaults to "output".

        Returns:
            str: Filename that imputed data was written to.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)

        nuc = {
            "A/A": "A",
            "G/G": "G",
            "C/C": "C",
            "T/T": "T",
            "A/G": "R",
            "G/A": "R",
            "C/T": "Y",
            "T/C": "Y",
            "G/C": "S",
            "C/G": "S",
            "A/T": "W",
            "T/A": "W",
            "G/T": "K",
            "T/G": "K",
            "A/C": "M",
            "C/A": "M",
        }

        df_decoded = df.copy()

        dreplace = dict()
        for col, ref, alt in zip(df.columns, self.ref, self.alt):

            ref2 = f"{ref}/{ref}"
            alt2 = f"{alt}/{alt}"
            het2 = f"{ref}/{alt}"

            if self.filetype.lower().startswith("phylip"):
                ref2 = nuc[ref2]
                alt2 = nuc[alt2]
                het2 = nuc[het2]

            d = {"0": ref2, 0: ref2, "1": het2, 1: het2, "2": alt2, 2: alt2}
            dreplace[col] = d

        df_decoded.replace(dreplace, inplace=True)

        ft = self.filetype.lower()

        if write_output:
            outfile = f"{prefix}_imputed"

        if ft.startswith("structure"):

            of = f"{outfile}.str"

            if ft.startswith("structure2row"):

                for col in df_decoded.columns:
                    df_decoded[col] = (
                        df_decoded[col]
                        .str.split("/")
                        .apply(lambda x: list(map(int, x)))
                    )

                df_decoded.insert(0, "sampleID", self.samples)
                df_decoded.insert(1, "popID", self.pops)

                df_decoded = (
                    df_decoded.set_index(["sampleID", "popID"])
                    .apply(pd.Series.explode)
                    .reset_index()
                )

            elif ft.startswith("structure1row"):
                df_decoded = pd.concat(
                    [
                        df_decoded[c]
                        .astype(str)
                        .str.split("/", expand=True)
                        .add_prefix(f"{c}_")
                        for c in df_decoded.columns
                    ],
                    axis=1,
                )

                df_decoded.insert(0, "sampleID", self.samples)
                df_decoded.insert(1, "popID", self.pops)

            if write_output:
                df_decoded.to_csv(
                    of,
                    sep="\t",
                    header=False,
                    index=False,
                )

        elif ft.startswith("phylip"):
            of = f"{outfile}.phy"
            header = f"{self.num_inds} {self.num_snps}\n"

            if write_output:
                with open(of, "w") as fout:
                    fout.write(header)

                lst_decoded = df_decoded.values.tolist()

                with open(of, "a") as fout:
                    for sample, row in zip(self.samples, lst_decoded):
                        seqs = "".join([str(x) for x in row])
                        fout.write(f"{sample}\t{seqs}\n")

                df_decoded.insert(0, "sampleID", self.samples)

        return of

    @property
    def snpcount(self) -> int:
        """Number of snps in the dataset.

        Returns:
            int: Number of SNPs per individual.
        """
        return self.num_snps

    @property
    def indcount(self) -> int:
        """Number of individuals in dataset.

        Returns:
            int: Number of individuals in input data.
        """
        return self.num_inds

    @property
    def populations(self) -> List[Union[str, int]]:
        """Population Ids.

        Returns:
            List[Union[str, int]]: Population IDs.
        """
        return self.pops

    @property
    def individuals(self) -> List[str]:
        """Sample IDs in input order.

        Returns:
            List[str]: Sample IDs in input order.
        """
        return self.samples

    @property
    def genotypes_list(self) -> List[List[int]]:
        """Encoded 012 genotypes as a 2D list.

        Returns:
            List[List[int]]: encoded 012 genotypes.
        """
        return self.snps

    @property
    def genotypes_nparray(self) -> np.ndarray:
        """012-encoded genotypes as a numpy.ndarray.

        Returns:
            numpy.ndarray of shape (n_samples, n_SNPs): 012-encoded genotypes of shape (n_samples, n_SNPs).
        """
        return np.array(self.snps)

    @property
    def genotypes_df(self) -> pd.DataFrame:
        """Encoded 012 genotypes as a pandas DataFrame object

        Returns:
            pandas.DataFrame of shape (n_samples, n_SNPs): 012-encoded genotypes.
        """
        df = pd.DataFrame.from_records(self.snps)
        df.replace(to_replace=-9.0, value=np.nan, inplace=True)
        return df.astype(np.float32)

    @property
    def genotypes_onehot(self) -> Union[np.ndarray, List[List[List[float]]]]:
        """One-hot encoded snps format.

        Returns:
            numpy.ndarray of shape (n_samples, n_SNPs): One-hot encoded numpy array.
        """
        return self.onehot


def merge_alleles(
    first: List[Union[str, int]],
    second: Optional[List[Union[str, int]]] = None,
) -> List[str]:
    """Merges first and second alleles in structure file.

    Args:
        first (List[Union[str, int] or None): Alleles on first line.
        second (List[Union[str, int]] or None, optional): Second row of alleles. Defaults to None.

    Returns:
        List[str]: VCF file-style genotypes (i.e. split by "/").

    Raises:
        ValueError: First and second lines have differing lengths.
        ValueError: Line has non-even number of alleles.
    """
    ret = list()
    if second is not None:
        if len(first) != len(second):
            raise ValueError(
                "First and second lines have different number of alleles\n"
            )
        else:
            for i in range(0, len(first)):
                ret.append(str(first[i]) + "/" + str(second[i]))
    else:
        if len(first) % 2 != 0:
            raise ValueError("Line has non-even number of alleles!\n")
        else:
            for i, j in zip(first[::2], first[1::2]):
                ret.append(str(i) + "/" + str(j))
    return ret
