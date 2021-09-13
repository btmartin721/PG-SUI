import os
import sys

# Make sure python version is >= 3.6
if sys.version_info < (3, 6):
    raise ImportError("Python < 3.6 is not supported!")

import numpy as np
import pandas as pd
import toytree as tt

from read_input.popmap_file import ReadPopmap
from utils import sequence_tools
from utils import settings


class GenotypeData:
    """[Class to read genotype and tree data and encode genotypes in 012 and onehot format

    Args:
            filename ([str]): [Path to input file containing genotypes]

            filetype ([str]): [Type of input genotype file. Possible ``filetype`` values include: 'phylip', 'structure1row', or 'structure2row'. VCF compatibility may be added in the future, but is not currently supported]

            popmapfile ([str]): [Path to population map file. If ``popmapfile`` is supplied and ``filetype`` is one of the STRUCTURE formats, then the structure file is assumed to have NO popID column]

            guidetree ([str]): [Path to input treefile]. Defaults to None.

            qmatrix_iqtree ([str]): [Path to iqtree output file containing q matrix]. Defaults to None.

            qmatrix ([str]): [Path to file containing only q matrix]. Defaults to None.

    Attributes:
            samples ([list(str)]): [List containing sample IDs of shape (n_samples,)]

            snps ([list(list(str))]): [2D list of shape (n_samples, n_sites) containing genotypes]

            pops ([list(str)]): [List of population IDs of shape (n_samples,)]

            onehot ([list(list(list(float)))]): [One-hot encoded genotypes as a 3D list of shape (n_samples, n_sites, 4). The inner-most list represents the four nucleotide bases in the order of 'A', 'T', 'G', 'C'. If position 0 contains a 1.0, then the site is an 'A'. If position 1 contains a 1.0, then the site is a 'T'...etc. Two values of 0.5 indicates a heterozygote. Missing data is encoded as four values of 0.0]

            guidetree ([toytree object]): [Input guide tree as a toytree object]

            num_snps ([int]): [Number of SNPs (features) present in the dataset]

            num_inds: ([int]): [Number of individuals (samples) present in the dataset]

    Properties:
            snpcount ([int]): [Number of SNPs (features) in the dataset]
            indcount ([int]): [Number of individuals (samples) in the dataset]
            populations ([list(str)]): [List of population IDs of shape (n_samples,)]

            individuals ([list(str)]): [List of sample IDs of shape (n_samples,)]
            genotypes_list ([list(list(str))]): [List of 012-encoded genotypes of shape (n_samples, n_sites)]

            genotypes_nparray ([numpy.ndarray]): [Numpy array of 012-encoded genotypes of shape (n_samples, n_sites)]

            genotypes_df ([pandas.DataFrame]): [Pandas DataFrame of 012-encoded genotypes of shape (n_samples, n_sites). Missing values are encoded as -9]

            genotypes_onehot ([numpy.ndarray(numpy.ndarray(numpy.ndarray)))]): [One-hot encoded numpy array (n_samples, n_sites, 4). The inner-most array consists of one-hot encoded values for the four nucleotides in the order of 'A', 'T', 'G', 'C'. Values of 0.5 indicate heterozygotes, and missing values contain 0.0 for all four nucleotides]
    """

    def __init__(
        self,
        filename=None,
        filetype=None,
        popmapfile=None,
        guidetree=None,
        qmatrix_iqtree=None,
        qmatrix=None,
    ):
        self.filename = filename
        self.filetype = filetype
        self.popmapfile = popmapfile
        self.guidetree = guidetree
        self.qmatrix_iqtree = qmatrix_iqtree
        self.qmatrix = qmatrix

        self.snpsdict = dict()
        self.samples = list()
        self.snps = list()
        self.pops = list()
        self.onehot = list()
        self.num_snps = 0
        self.num_inds = 0

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

    def parse_filetype(self, filetype=None, popmapfile=None):
        if filetype is None:
            raise OSError("No filetype specified.\n")
        else:
            if filetype == "phylip":
                self.filetype = filetype
                self.read_phylip()
            elif filetype == "structure1row":
                if popmapfile is not None:
                    self.filetype = "structure1row"
                    self.read_structure(onerow=True, popids=False)
                else:
                    self.filetype = "structure1rowPopID"
                    self.read_structure(onerow=True, popids=True)
            elif filetype == "structure2row":
                if popmapfile is not None:
                    self.filetype = "structure2row"
                    self.read_structure(onerow=False, popids=False)
                else:
                    self.filetype = "structure2rowPopID"
                    self.read_structure(onerow=False, popids=True)
            else:
                raise OSError(f"Filetype {filetype} is not supported!\n")

    def check_filetype(self, filetype):
        if self.filetype is None:
            self.filetype = filetype
        elif self.filetype == filetype:
            pass
        else:
            raise TypeError(
                "GenotypeData read_XX() call does not match filetype!\n"
            )

    def read_tree(self, treefile):
        """[Read Newick-style phylogenetic tree into toytree object. Format should be of type 0 (see toytree documentation)]

        Args:
                treefile ([str]): [Path to Newick-style tree file]

        Returns:
                [toytree object]: [Tree as toytree object]
        """
        if not os.path.isfile(treefile):
            raise FileNotFoundError(f"File {treefile} not found!")

        assert os.access(treefile, os.R_OK), f"File {treefile} isn't readable"

        return tt.tree(treefile, tree_format=0)

    def q_from_file(self, fname, label=True):
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

    def q_from_iqtree(self, iqfile):
        """[Read in q-matrix from *.iqtree file. The *.iqtree file is output when running IQ-TREE and contains the standard output of the IQ-TREE run]

        Args:
                iqfile ([str]): [Path to *.iqtree file]

        Returns:
                [pandas.DataFrame]: [Q-matrix as pandas DataFrame]

        Raises:
                FileNotFoundError: [If iqtree file could not be found]
                IOError: [If iqtree file could not be read from]
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

    def blank_q_matrix(self, default=0.0):
        q = dict()
        for nuc1 in ["A", "C", "G", "T"]:
            q[nuc1] = dict()
            for nuc2 in ["A", "C", "G", "T"]:
                q[nuc1][nuc2] = default
        return q

    def read_structure(self, onerow=False, popids=True):
        """[Read a structure file with two rows per individual]"""
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
                    if not firstline:
                        firstline = line.split()
                        continue
                    else:
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

        print("Done!")

        print("\nConverting genotypes to one-hot encoding...")
        # Convert snp_data to onehot encoding format
        self.convert_onehot(snp_data)

        print("Done!")

        print("\nConverting genotypes to 012 format...")
        # Convert snp_data to 012 format

        self.convert_012(snp_data, vcf=True)

        print("Done!")

        # Get number of samples and snps
        self.num_snps = len(self.snps[0])
        self.num_inds = len(self.samples)

        print(
            f"\nFound {self.num_snps} SNPs and {self.num_inds} individuals...\n"
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

    def read_phylip(self):
        """[Populates ReadInput object by parsing Phylip]

        Args:
                popmap_filename [str]: [Filename for population map file]
        """
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

        print("Done!")

        print("\nConverting genotypes to one-hot encoding...")
        # Convert snp_data to onehot format
        self.convert_onehot(snp_data)
        print("Done!")

        print("\nConverting genotypes to 012 encoding...")
        # Convert snp_data to 012 format

        self.convert_012(snp_data)
        print("Done!")

        self.num_snps = num_snps
        self.num_inds = num_inds

        # Error handling if incorrect number of individuals in header.
        if len(self.samples) != num_inds:
            raise ValueError(
                "Incorrect number of individuals listed in header\n"
            )

    def read_phylip_tree_imputation(self, aln):
        """[Function to read a alignment file. Returns dict (key=sample) of lists (sequences divided by site; i.e., all sites for one sample across all columns)]

        Args:
                aln ([str]): [Path to alignment file]

        Raises:
                IOError: [Raise exception if alignment file could not be read from]
                FileNotFoundError: [Raise exception if alignment file not found]
        """
        if phy is None:
            raise TypeError(
                "alignment file must be specified if using PHYLIP input format, "
                "but got NoneType"
            )

        elif os.path.exists(phy):
            with open(phy, "r") as fh:
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
                    print(f"Could not read file {phy}")
                    sys.exit(1)
                finally:
                    fh.close()
        else:
            raise FileNotFoundError(f"File {phy} not found!")

    def convert_012(self, snps, vcf=False, impute_mode=False):
        """[Encode IUPAC nucleotides as 0 (reference), 1 (heterogygous), and 2 (alternate) alleles]

        Args:
                snps ([list(list(str))]): [2D list of genotypes of shape (n_samples, n_sites)]

                vcf (bool, optional): [Whether or not VCF file input is provided]. Defaults to False.

                impute_mode (bool, optional): [Whether or not convert_012() is called in impute mode. If True, then returns the 012-encoded genotypes and does not set the ``self.snps`` attribute. If False, it does the opposite]. Defaults to False.

        Returns:
                (list(list(int)), optional): [012-encoded genotypes as a 2D list of shape (n_samples, n_sites). Only returns value if ``impute_mode`` is True]

                (list(int)), optional): [List of integers indicating bi-allelic site indexes]
        """
        skip = 0
        new_snps = list()

        if impute_mode:
            imp_snps = list()

        for i in range(0, len(snps)):
            new_snps.append([])

        valid_sites = np.ones(len(snps[0]))
        for j in range(0, len(snps[0])):
            loc = list()
            for i in range(0, len(snps)):
                if vcf:
                    loc.append(snps[i][j])
                else:
                    loc.append(snps[i][j].upper())

            if sequence_tools.count_alleles(loc, vcf=vcf) != 2:
                skip += 1
                valid_sites[j] = np.nan
                continue
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
            return imp_snps, valid_sites

    def convert_onehot(self, snp_data, encodings_dict=None):

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
            self.filetype == "structure1row"
            or self.filetype == "structure2row"
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

    def read_popmap(self, popmapfile):
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
            sys.exit(
                "Error: The number of individuals in the popmap file differs from the number of sequences\n"
            )

        for sample in self.samples:
            if sample in my_popmap:
                self.pops.append(my_popmap[sample])

    @property
    def snpcount(self):
        """[Getter for number of snps in the dataset]

        Returns:
                [int]: [Number of SNPs per individual]
        """
        return self.num_snps

    @property
    def indcount(self):
        """[Getter for number of individuals in dataset]

        Returns:
                [int]: [Number of individuals in input sequence data]
        """
        return self.num_inds

    @property
    def populations(self):
        """[Getter for population IDs]

        Returns:
                [list]: [Poulation IDs as a list]
        """
        return self.pops

    @property
    def individuals(self):
        """[Getter for sample IDs in input order]

        Returns:
                [list]: [sample IDs as a list in input order]
        """
        return self.samples

    @property
    def genotypes_list(self):
        """[Getter for the 012 genotypes]

        Returns:
                [list(list)]: [012 genotypes as a 2d list]
        """
        return self.snps

    @property
    def genotypes_nparray(self):
        """[Getter for 012 genotypes as a numpy array]

        Returns:
                [2D numpy.array]: [012 genotypes as shape (n_samples, n_variants)]
        """
        return np.array(self.snps)

    @property
    def genotypes_df(self):
        """[Getter for 012 genotypes as a pandas DataFrame object]

        Returns:
                [pandas.DataFrame]: [012-encoded genotypes as pandas DataFrame]
        """
        df = pd.DataFrame.from_records(self.snps)
        df.replace(to_replace=-9, value=np.nan, inplace=True)
        return df.astype(np.float32)

    @property
    def genotypes_onehot(self):
        """[Getter for one-hot encoded snps format]

        Returns:
                [2D numpy.array]: [One-hot encoded numpy array (n_samples, n_variants)]
        """
        return self.onehot

    def set_tree(self):
        pass
        # self.


def merge_alleles(first, second=None):
    """[Merges first and second alleles in structure file]

    Args:
            first ([list]): [Alleles on one line]
            second ([list], optional): [Second row of alleles]. Defaults to None.

    Returns:
            [list(str)]: [VCF-style genotypes (i.e. split by "/")]
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


def count2onehot(samples, snps):
    onehot_dict = {
        "0": [1.0, 0.0],
        "1": [0.5, 0.5],
        "2": [0.0, 1.0],
        "-": [0.0, 0.0],
    }
    onehot_outer_list = list()
    for i in range(len(samples)):
        onehot_list = list()
        for j in range(len(snps[0])):
            onehot_list.append(onehot_dict[snps[i][j]])
        onehot_outer_list.append(onehot_list)
    onehot = np.array(onehot_outer_list)
    return onehot
