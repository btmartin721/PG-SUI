import os
import sys
from pathlib import Path
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any, Callable
from copy import deepcopy

# Third-party imports
import numpy as np
import pandas as pd
import scipy.linalg
import toyplot.pdf
import toyplot as tp
import toytree as tt
from decimal import Decimal

from sklearn.impute import SimpleImputer

# Custom imports
try:
    from snpio import GenotypeData
    from ..utils.misc import isnotebook
except (ModuleNotFoundError, ValueError, ImportError):
    from snpio import GenotypeData
    from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar

# Pandas on pip gives a performance warning when doing the below code.
# Apparently it's a bug that exists in the pandas version I used here.
# It can be safely ignored.
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class ImputePhylo:
    """Impute missing data using a phylogenetic tree to inform the imputation.

    Args:
        genotype_data (GenotypeData instance): GenotypeData instance. Must have the q, tree, and optionally site_rates attributes defined.

        minbr (float or None, optional): Minimum branch length. Defaults to 0.0000000001

        str_encodings (Dict[str, int], optional): Integer encodings used in STRUCTURE-formatted file. Should be a dictionary with keys=nucleotides and values=integer encodings. The missing data encoding should also be included. Argument is ignored if using a PHYLIP-formatted file. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}

        prefix (str, optional): Prefix to use with output files.

        save_plots (bool, optional): Whether to save PDF files with genotype imputations for each site to disk. It makes one PDF file per locus, so if you have a lot of loci it will make a lot of PDF files. Defaults to False.

        write_output (bool, optional): Whether to save the imputed data to disk. Defaults to True.

        disable_progressbar (bool, optional): Whether to disable the progress bar during the imputation. Defaults to False.

        kwargs (Dict[str, Any] or None, optional): Additional keyword arguments intended for internal purposes only. Possible arguments: {"column_subset": List[int] or numpy.ndarray[int]}; Subset SNPs by a list of indices for IterativeImputer. Defauls to None.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure",
        >>>    popmapfile="test.popmap",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree",
        >>>    siterates_iqtree="test.rates",
        >>>)
        >>>
        >>>phylo = ImputePhylo(
        >>>     genotype_data=data,
        >>>     save_plots=True,
        >>>)
        >>> # Get GenotypeData object.
        >>>gd_phylo = phylo.imputed
    """

    def __init__(
        self,
        genotype_data: Optional[Any],
        minbr: Optional[float] = 0.0000000001,
        *,
        str_encodings: Dict[str, int] = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "N": -9,
        },
        prefix: str = "output",
        save_plots: bool = False,
        disable_progressbar: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        self.genotype_data = genotype_data
        self.alnfile = genotype_data.filename
        self.filetype = genotype_data.filetype
        self.popmap = genotype_data.popmap
        self.str_encodings = str_encodings
        self.prefix = prefix
        self.minbr = minbr
        self.save_plots = save_plots
        self.disable_progressbar = disable_progressbar
        self.column_subset = kwargs.get("column_subset", None)
        self.validation_mode = kwargs.get("validation_mode", False)

        self.valid_sites = None
        self.valid_sites_count = None

        self._validate_arguments(genotype_data)
        data, tree, q, site_rates = self._parse_arguments(genotype_data)

        if not self.validation_mode:
            imputed012 = self.impute_phylo(tree, data, q, site_rates)
            genotype_data = deepcopy(genotype_data)
            genotype_data.genotypes_012 = imputed012
            self.imputed = genotype_data
        else:
            self.imputed = self.impute_phylo(tree, data, q, site_rates)

    @property
    def genotypes_012(self):
        return self.imputed.genotypes012

    @property
    def snp_data(self):
        return self.imputed.snp_data

    @property
    def alignment(self):
        return self.imputed.alignment

    def impute_phylo(
        self,
        tree: tt.tree,
        genotypes: Dict[str, List[Union[str, int]]],
        Q: pd.DataFrame,
        site_rates=None,
        minbr=0.0000000001,
    ) -> pd.DataFrame:
        """Imputes genotype values with a guide tree.

        Imputes genotype values by using a provided guide
        tree to inform the imputation, assuming maximum parsimony.

        Process Outline:
            For each SNP:
            1) if site_rates, get site-transformated Q matrix.

            2) Postorder traversal of tree to compute ancestral
            state likelihoods for internal nodes (tips -> root).
            If exclude_N==True, then ignore N tips for this step.

            3) Preorder traversal of tree to populate missing genotypes
            with the maximum likelihood state (root -> tips).

        Args:
            tree (toytree.tree object): Input tree.

            genotypes (Dict[str, List[Union[str, int]]]): Dictionary with key=sampleids, value=sequences.

            Q (pandas.DataFrame): Rate Matrix Q from .iqtree or separate file.

            site_rates (List): Site-specific substitution rates (used to weight per-site Q)

            minbr (float) : Minimum branch length (those below this value will be treated as == minbr)

        Returns:
            pandas.DataFrame: Imputed genotypes.

        Raises:
            IndexError: If index does not exist when trying to read genotypes.
            AssertionError: Sites must have same lengths.
            AssertionError: Missing data still found after imputation.
        """
        try:
            if list(genotypes.values())[0][0][1] == "/":
                genotypes = self._str2iupac(genotypes, self.str_encodings)
        except IndexError:
            if self._is_int(list(genotypes.values())[0][0][0]):
                raise

        if self.column_subset is not None:
            if isinstance(self.column_subset, np.ndarray):
                self.column_subset = self.column_subset.tolist()

            genotypes = {
                k: [v[i] for i in self.column_subset]
                for k, v in genotypes.items()
            }

        # For each SNP:
        nsites = list(set([len(v) for v in genotypes.values()]))
        assert len(nsites) == 1, "Some sites have different lengths!"

        outdir = f"{self.prefix}_imputation_plots"

        if self.save_plots:
            Path(outdir).mkdir(parents=True, exist_ok=True)

        for snp_index in progressbar(
            range(nsites[0]),
            desc="Feature Progress: ",
            leave=True,
            disable=self.disable_progressbar,
        ):
            rate = 1.0
            if site_rates is not None:
                rate = site_rates[snp_index]

            site_Q = Q.copy(deep=True) * rate

            bads = list()
            for samp in genotypes.keys():
                if genotypes[samp][snp_index].upper() == "N":
                    bads.append(samp)

            # postorder traversal to compute likelihood at root
            node_lik = dict()
            for node in tree.treenode.traverse("postorder"):
                if node.is_leaf():
                    continue

                if node.idx not in node_lik:
                    node_lik[node.idx] = [1.0, 1.0, 1.0, 1.0]

                for child in node.get_children():
                    # get branch length to child
                    # bl = child.edge.length
                    # get transition probs
                    d = child.dist
                    if d < minbr:
                        d = minbr
                    pt = self._transition_probs(site_Q, d)
                    if child.is_leaf():
                        if child.name in genotypes:
                            if child.name in bads:
                                sum = [1.0, 1.0, 1.0, 1.0]
                            else:
                                # get genotype data
                                sum = None
                                for allele in self._get_iupac_full(
                                    genotypes[child.name][snp_index]
                                ):
                                    if sum is None:
                                        sum = [
                                            Decimal(x)
                                            for x in list(pt[allele])
                                        ]
                                    else:
                                        sum = [
                                            Decimal(sum[i]) + Decimal(val)
                                            for i, val in enumerate(
                                                list(pt[allele])
                                            )
                                        ]
                            node_lik[child.idx] = [Decimal(x) for x in sum]

                            # add to likelihood for parent node
                            if node_lik[node.idx] is None:
                                node_lik[node.idx] = node_lik[child.idx]
                            else:
                                node_lik[node.idx] = [
                                    Decimal(node_lik[child.idx][i])
                                    * Decimal(val)
                                    for i, val in enumerate(node_lik[node.idx])
                                ]
                        else:
                            # raise error
                            sys.exit(
                                f"Error: Taxon {child.name} not found in "
                                f"genotypes"
                            )
                    else:
                        l = self._get_internal_lik(pt, node_lik[child.idx])
                        if node_lik[node.idx] is None:
                            node_lik[node.idx] = [Decimal(x) for x in l]

                        else:
                            node_lik[node.idx] = [
                                Decimal(l[i]) * Decimal(val)
                                for i, val in enumerate(node_lik[node.idx])
                            ]

            # preorder traversal to get marginal reconstructions at internal 
            # nodes
            marg = node_lik.copy()
            for node in tree.treenode.traverse("preorder"):
                if node.is_root():
                    continue
                elif node.is_leaf():
                    continue
                lik_arr = marg[node.idx]
                parent_arr = marg[node.up.idx]
                marg[node.idx] = [
                    Decimal(lik) * (Decimal(parent_arr[i]) / Decimal(lik))
                    for i, lik in enumerate(lik_arr)
                ]

            # get marginal reconstructions for bad bois
            two_pass = dict()
            for samp in bads:
                # get most likely state for focal tip
                node = tree.idx_dict[
                    tree.get_mrca_idx_from_tip_labels(names=samp)
                ]
                dist = node.dist
                parent = node.up
                imputed = None
                pt = self._transition_probs(site_Q, dist)
                lik = self._get_internal_lik(pt, marg[parent.idx])

                tol = 0.001
                imputed = self._get_imputed_nuc(lik)

                # two_pass[samp] = [imputed, lik]
                genotypes[samp][snp_index] = imputed

            # DEPRECATED: RE-ROOTING METHOD OF YANG ET AL
            # NEW METHOD (ABOVE) IS LINEAR
            # reroot=dict()
            # for samp in bads:
            #     #focaltree = tree.drop_tips(names=[x for x in bads if x != samp])
            #     focaltree = tree.root(names=[samp])
            #
            #     mystyle = {
            #         "edge_type": "p",
            #         "edge_style": {
            #             "stroke-width": 1,
            #         },
            #         "tip_labels_align": True,
            #         "tip_labels_style": {"font-size": "5px"},
            #         "node_labels": False,
            #     }
            #
            #     canvas, axes, mark = focaltree.draw()
            #     toyplot.pdf.render(canvas, "test.pdf")
            #
            #     #postorder traversal to compute likelihood
            #     node_lik = dict()
            #     for node in focaltree.treenode.traverse("postorder"):
            #         if node.is_leaf():
            #             continue
            #
            #         if node.idx not in node_lik:
            #             node_lik[node.idx] = None
            #
            #         for child in node.get_children():
            #             # get branch length to child
            #             # bl = child.edge.length
            #             # get transition probs
            #             pt = self._transition_probs(site_Q, child.dist)
            #             if child.is_leaf():
            #                 if child.name in genotypes:
            #                     if child.name in bads:
            #                         sum = [1.0, 1.0, 1.0, 1.0]
            #                     else:
            #                         # get genotype data
            #                         sum = None
            #                         for allele in self._get_iupac_full(
            #                             genotypes[child.name][snp_index]
            #                         ):
            #                             if sum is None:
            #                                 sum = [Decimal(x) for x in list(pt[allele])]
            #                             else:
            #                                 sum = [
            #                                     Decimal(sum[i]) + Decimal(val)
            #                                     for i, val in enumerate(
            #                                         list(pt[allele])
            #                                     )
            #                                     ]
            #
            #                     node_lik[child.idx] = [Decimal(x) for x in sum]
            #
            #                     #add to likelihood for parent node
            #                     if node_lik[node.idx] is None:
            #                         node_lik[node.idx] = node_lik[child.idx]
            #                     else:
            #                         node_lik[node.idx] = [
            #                             Decimal(node_lik[child.idx][i]) * Decimal(val)
            #                             for i, val in enumerate(node_lik[node.idx])
            #                         ]
            #                 else:
            #                     # raise error
            #                     sys.exit(
            #                         f"Error: Taxon {child.name} not found in "
            #                         f"genotypes"
            #                     )
            #             else:
            #                 l = self._get_internal_lik(pt, node_lik[child.idx])
            #                 if node_lik[node.idx] is None:
            #                     node_lik[node.idx] = [Decimal(x) for x in l]
            #
            #                 else:
            #                     node_lik[node.idx] = [
            #                         Decimal(l[i]) * Decimal(val)
            #                         for i, val in enumerate(node_lik[node.idx])
            #                     ]
            #
            #     # get most likely state for focal tip
            #     node = focaltree.idx_dict[
            #         focaltree.get_mrca_idx_from_tip_labels(names=samp)
            #     ]
            #     dist = node.dist
            #     parent = node.up
            #     imputed = None
            #     pt = self._transition_probs(site_Q, dist)
            #     lik = self._get_internal_lik(pt, node_lik[parent.idx])
            #     maxpos = lik.index(max(lik))
            #     if maxpos == 0:
            #         imputed = "A"
            #
            #     elif maxpos == 1:
            #         imputed = "C"
            #
            #     elif maxpos == 2:
            #         imputed = "G"
            #
            #     else:
            #         imputed = "T"
            #     reroot[samp] = [imputed, lik]
            # check if two methods give same results
            # for key in two_pass:
            #     if two_pass[key][0] != reroot[key][0]:
            #         print("Two-pass:", two_pass[key][0], "-", two_pass[key][1])
            #         print("Reroot:", reroot[key][0], "-", reroot[key][1])

            if self.save_plots:
                self._draw_imputed_position(
                    tree,
                    bads,
                    genotypes,
                    snp_index,
                    f"{outdir}/{self.prefix}_pos{snp_index}.pdf",
                )

        df = pd.DataFrame.from_dict(genotypes, orient="index")

        # Make sure no missing data remains in the dataset
        assert (
            not df.isin([-9]).any().any()
        ), "Imputation failed...Missing values found in the imputed dataset"

        (
            imp_snps,
            self.valid_sites,
            self.valid_sites_count,
        ) = self.genotype_data.convert_012(
            df.to_numpy().tolist(), impute_mode=True
        )

        df_imp = pd.DataFrame.from_records(imp_snps)

        return df_imp

    def nbiallelic(self) -> int:
        """Get the number of remaining bi-allelic sites after imputation.

        Returns:
            int: Number of bi-allelic sites remaining after imputation.
        """
        return len(self.imputed.columns)

    def _get_imputed_nuc(self, lik_arr):
        nucmap = {0: "A", 1: "C", 2: "G", 3: "T"}
        maxpos = lik_arr.index(max(lik_arr))
        picks = set([maxpos])
        # NOT USED:
        # Experimenting with ways to impute heterozygotes.
        # Note that LRT isn't appropriate (as I used here) because
        # the models are not nested & LRTS isn't necessarily expected
        # to be chisq distributed.
        # Check out Vuong test and read Lewis et al 2011 (doi: 10.1111/j.2041-210X.2010.00063.x)
        #
        # for index, alt in enumerate(lik_arr):
        #     if index == maxpos:
        #         continue
        #     else:
        #         lr = lrt(lik_arr[maxpos], alt, loglik=False)
        #         p = chi2.sf(lr)
        #         print(nucmap[maxpos], ":", str(lrt(lik_arr[maxpos], alt, loglik=False)), p)
        return nucmap[maxpos]

    def _parse_arguments(
        self, genotype_data: Any
    ) -> Tuple[Dict[str, List[Union[int, str]]], tt.tree, pd.DataFrame]:
        """Determine which arguments were specified and set appropriate values.

        Args:
            genotype_data (GenotypeData object): Initialized GenotypeData object.

        Returns:
            Dict[str, List[Union[int, str]]]: GenotypeData.snpsdict object. If genotype_data is not None, then this value gets set from the GenotypeData.snpsdict object. If alnfile is not None, then the alignment file gets read and the snpsdict object gets set from the alnfile.

            toytree.tree: Input phylogeny, either read from GenotypeData object or supplied with treefile.

            pandas.DataFrame: Q Rate Matrix, either from IQ-TREE file or from its own supplied file.
        """
        data = genotype_data.snpsdict
        tree = genotype_data.tree

        # read (optional) Q-matrix
        if genotype_data.q is not None:
            q = genotype_data.q
        else:
            raise TypeError("q must be defined in GenotypeData instance.")

        # read (optional) site-specific substitution rates
        if genotype_data.site_rates is not None:
            site_rates = genotype_data.site_rates
        else:
            raise TypeError(
                "site rates must be defined in GenotypeData instance."
            )

        return data, tree, q, site_rates

    def _validate_arguments(self, genotype_data: Any) -> None:
        """Validate that the correct arguments were supplied.

        Args:
            genotype_data (GenotypeData object): Input GenotypeData instance.

        Raises:
            TypeError: Must define genotype_data.tree in GenotypeData instance.
            TypeError: Q rate matrix must be defined in GenotypeData instance.
        """

        if genotype_data.tree is None:
            raise TypeError("genotype_data.tree must be defined")

        if genotype_data.q is None:
            raise TypeError("q must be defined in GenotypeData instance.")

    def _print_q(self, q: pd.DataFrame) -> None:
        """Print Rate Matrix Q.

        Args:
            q (pandas.DataFrame): Rate Matrix Q.
        """
        print("Rate matrix Q:")
        print("\tA\tC\tG\tT\t")
        for nuc1 in ["A", "C", "G", "T"]:
            print(nuc1, end="\t")
            for nuc2 in ["A", "C", "G", "T"]:
                print(q[nuc1][nuc2], end="\t")
            print("")

    def _is_int(self, val: Union[str, int]) -> bool:
        """Check if value is integer.

        Args:
            val (int or str): Value to check.

        Returns:
            bool: True if integer, False if string.
        """
        try:
            num = int(val)
        except ValueError:
            return False
        return True

    def _get_nuc_colors(self, nucs: List[str]) -> List[str]:
        """Get colors for each nucleotide when plotting.

        Args:
            nucs (List[str]): Nucleotides at current site.

        Returns:
            List[str]: Hex-code color values for each IUPAC nucleotide.
        """
        ret = list()
        for nuc in nucs:
            nuc = nuc.upper()
            if nuc == "A":
                ret.append("#0000FF")  # blue
            elif nuc == "C":
                ret.append("#FF0000")  # red
            elif nuc == "G":
                ret.append("#00FF00")  # green
            elif nuc == "T":
                ret.append("#FFFF00")  # yellow
            elif nuc == "R":
                ret.append("#0dbaa9")  # blue-green
            elif nuc == "Y":
                ret.append("#FFA500")  # orange
            elif nuc == "K":
                ret.append("#9acd32")  # yellow-green
            elif nuc == "M":
                ret.append("#800080")  # purple
            elif nuc == "S":
                ret.append("#964B00")
            elif nuc == "W":
                ret.append("#C0C0C0")
            else:
                ret.append("#000000")
        return ret

    def _label_bads(
        self, tips: List[str], labels: List[str], bads: List[str]
    ) -> List[str]:
        """Insert asterisks around bad nucleotides.

        Args:
            tips (List[str]): Tip labels (sample IDs).
            labels (List[str]): List of nucleotides at current site.
            bads (List[str]): List of tips that have missing data at current site.

        Returns:
            List[str]: IUPAC Nucleotides with "*" inserted around tips that had missing data.
        """
        for i, t in enumerate(tips):
            if t in bads:
                labels[i] = "*" + str(labels[i]) + "*"
        return labels

    def _draw_imputed_position(
        self,
        tree: tt.tree,
        bads: List[str],
        genotypes: Dict[str, List[str]],
        pos: int,
        out: str = "tree.pdf",
    ) -> None:
        """Draw nucleotides at phylogeny tip and saves to file on disk.

        Draws nucleotides as tip labels for the current SNP site. Imputed values have asterisk surrounding the nucleotide label. The tree is converted to a toyplot object and saved to file.

        Args:
            tree (toytree.tree): Input tree object.
            bads (List[str]): List of sampleIDs that have missing data at the current SNP site.
            genotypes (Dict[str, List[str]]): Genotypes at all SNP sites.
            pos (int): Current SNP index.
            out (str, optional): Output filename for toyplot object.
        """

        # print(tree.get_tip_labels())
        sizes = [8 if i in bads else 0 for i in tree.get_tip_labels()]
        colors = [genotypes[i][pos] for i in tree.get_tip_labels()]
        labels = colors

        labels = self._label_bads(tree.get_tip_labels(), labels, bads)

        colors = self._get_nuc_colors(colors)

        mystyle = {
            "edge_type": "p",
            "edge_style": {
                "stroke": tt.colors[0],
                "stroke-width": 1,
            },
            "tip_labels_align": True,
            "tip_labels_style": {"font-size": "5px"},
            "node_labels": False,
        }

        canvas, axes, mark = tree.draw(
            tip_labels_colors=colors,
            tip_labels=labels,
            width=400,
            height=600,
            **mystyle,
        )

        toyplot.pdf.render(canvas, out)

    def _all_missing(
        self,
        tree: tt.tree,
        node_index: int,
        snp_index: int,
        genotypes: Dict[str, List[str]],
    ) -> bool:
        """Check if all descendants of a clade have missing data at SNP site.

        Args:
            tree (toytree.tree): Input guide tree object.

            node_index (int): Parent node to determine if all descendants have missing data.

            snp_index (int): Index of current SNP site.

            genotypes (Dict[str, List[str]]): Genotypes at all SNP sites.

        Returns:
            bool: True if all descendants have missing data, otherwise False.
        """
        for des in tree.get_tip_labels(idx=node_index):
            if genotypes[des][snp_index].upper() not in ["N", "-"]:
                return False
        return True

    def _get_internal_lik(
        self, pt: pd.DataFrame, lik_arr: List[float]
    ) -> List[float]:
        """Get ancestral state likelihoods for internal nodes of the tree.

        Postorder traversal to calculate internal ancestral state likelihoods (tips -> root).

        Args:
            pt (pandas.DataFrame): Transition probabilities calculated from Rate Matrix Q.
            lik_arr (List[float]): Likelihoods for nodes or leaves.

        Returns:
            List[float]: Internal likelihoods.
        """
        ret = list()
        for i, val in enumerate(lik_arr):
            col = list(pt.iloc[:, i])
            sum = Decimal(0.0)
            for v in col:
                sum += Decimal(v) * Decimal(val)
            ret.append(sum)
        return ret

    def _transition_probs(self, Q: pd.DataFrame, t: float) -> pd.DataFrame:
        """Get transition probabilities for tree.

        Args:
            Q (pd.DataFrame): Rate Matrix Q.
            t (float): Tree distance of child.

        Returns:
            pd.DataFrame: Transition probabilities.
        """
        ret = Q.copy(deep=True)
        m = Q.to_numpy()
        pt = scipy.linalg.expm(m * t)
        ret[:] = pt
        return ret

    def _str2iupac(
        self, genotypes: Dict[str, List[str]], str_encodings: Dict[str, int]
    ) -> Dict[str, List[str]]:
        """Convert STRUCTURE-format encodings to IUPAC bases.

        Args:
            genotypes (Dict[str, List[str]]): Genotypes at all sites.
            str_encodings (Dict[str, int]): Dictionary that maps IUPAC bases (keys) to integer encodings (values).

        Returns:
            Dict[str, List[str]]: Genotypes converted to IUPAC format.
        """
        a = str_encodings["A"]
        c = str_encodings["C"]
        g = str_encodings["G"]
        t = str_encodings["T"]
        n = str_encodings["N"]
        nuc = {
            f"{a}/{a}": "A",
            f"{c}/{c}": "C",
            f"{g}/{g}": "G",
            f"{t}/{t}": "T",
            f"{n}/{n}": "N",
            f"{a}/{c}": "M",
            f"{c}/{a}": "M",
            f"{a}/{g}": "R",
            f"{g}/{a}": "R",
            f"{a}/{t}": "W",
            f"{t}/{a}": "W",
            f"{c}/{g}": "S",
            f"{g}/{c}": "S",
            f"{c}/{t}": "Y",
            f"{t}/{c}": "Y",
            f"{g}/{t}": "K",
            f"{t}/{g}": "K",
        }

        for k, v in genotypes.items():
            for i, gt in enumerate(v):
                v[i] = nuc[gt]

        return genotypes

    def _get_iupac_full(self, char: str) -> List[str]:
        """Map nucleotide to list of expanded IUPAC encodings.

        Args:
            char (str): Current nucleotide.

        Returns:
            List[str]: List of nucleotides in ``char`` expanded IUPAC.
        """
        char = char.upper()
        iupac = {
            "A": ["A"],
            "G": ["G"],
            "C": ["C"],
            "T": ["T"],
            "N": ["A", "C", "T", "G"],
            "-": ["A", "C", "T", "G"],
            "R": ["A", "G"],
            "Y": ["C", "T"],
            "S": ["G", "C"],
            "W": ["A", "T"],
            "K": ["G", "T"],
            "M": ["A", "C"],
            "B": ["C", "G", "T"],
            "D": ["A", "G", "T"],
            "H": ["A", "C", "T"],
            "V": ["A", "C", "G"],
        }

        ret = iupac[char]
        return ret


class ImputeAlleleFreq:
    """Impute missing data by global allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance.

        by_populations (bool, optional): Whether or not to impute by-population or globally. Defaults to False (global allele frequency).

        diploid (bool, optional): When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency. Defaults to True.

        default (int, optional): Value to set if no alleles sampled at a locus. Defaults to 0.

        missing (int, optional): Missing data value. Defaults to -9.

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool, validation_mode: bool, gt: List[List[int]]}. "iterative_mode" determines whether ``ImputeAlleleFreq`` is being used as the initial imputer in ``IterativeImputer``\. ``gt`` is used internally for the simple imputers during grid searches and validation. If ``genotype_data is None`` then ``gt`` cannot also be None, and vice versa. Only one of ``gt`` or ``genotype_data`` can be set.

    Raises:
        TypeError: genotype_data cannot be NoneType.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    popmapfile="test.popmap",
        >>>)
        >>>
        >>>afpop = ImputeAlleleFreq(
        >>>     genotype_data=data,
        >>>     by_populations=True,
        >>>)
        >>>
        >>>gd_afpop = afpop.imputed
    """

    def __init__(
        self,
        genotype_data: GenotypeData,
        *,
        by_populations: bool = False,
        diploid: bool = True,
        default: int = 0,
        missing: int = -9,
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        if genotype_data is None and gt is None:
            raise TypeError("GenotypeData instance or gt must be provided.")

        gt = kwargs.get("gt", None)

        if gt is None:
            gt_list = genotype_data.genotypes_012(fmt="list")
        else:
            gt_list = gt

        if by_populations:
            if genotype_data.populations is None:
                raise TypeError(
                    "When by_populations is True, GenotypeData instance must have a defined populations attribute"
                )

            self.pops = genotype_data.populations

        else:
            self.pops = None

        self.diploid = diploid
        self.default = default
        self.missing = missing
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)
        self.validation_mode = kwargs.get("validation_mode", False)

        if not self.validation_mode:
            imputed012, self.valid_cols = self.fit_predict(gt_list)
            genotype_data = deepcopy(genotype_data)
            genotype_data.genotypes_012 = imputed012
            self.imputed = genotype_data
        else:
            self.imputed, self.valid_cols = self.fit_predict(gt_list)

    @property
    def genotypes_012(self):
        return self.imputed.genotypes_012

    @property
    def snp_data(self):
        return self.imputed.snp_data

    @property
    def alignment(self):
        return self.imputed.alignment

    def fit_predict(
        self, X: List[List[int]]
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]],
        List[int],
    ]:
        """Impute missing genotypes using allele frequencies.

        Impute using global or by_population allele frequencies. Missing alleles are primarily coded as negative; usually -9.

        Args:
            X (List[List[int]], numpy.ndarray, or pandas.DataFrame): 012-encoded genotypes obtained from the GenotypeData object.

        Returns:
            pandas.DataFrame, numpy.ndarray, or List[List[Union[int, float]]]: Imputed genotypes of same shape as data.

            List[int]: Column indexes that were retained.

        Raises:
            TypeError: X must be either list, np.ndarray, or pd.DataFrame.
        """
        if self.pops is not None and self.verbose:
            print("\nImputing by population allele frequencies...")
        elif self.pops is None and self.verbose:
            print("\nImputing by global allele frequency...")

        if isinstance(X, (list, np.ndarray)):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(
                f"X must be of type list(list(int)), numpy.ndarray, "
                f"or pandas.DataFrame, but got {type(X)}"
            )

        df = df.astype(int)
        df.replace(self.missing, np.nan, inplace=True)

        # Initialize an empty list to hold the columns
        columns = []
        valid_cols = list()
        bad_cnt = 0

        if self.pops is not None:
            df = df.copy()

            # Impute per-population mode.
            df["pops"] = self.pops
            groups = df.groupby(["pops"], sort=False)

            for col in df.columns:
                try:
                    # Instead of appending to the DataFrame, append to the list
                    columns.append(
                        groups[col].transform(
                            lambda x: x.fillna(x.mode().iloc[0])
                        )
                    )

                    if col != "pops":
                        valid_cols.append(col)

                except IndexError as e:
                    if str(e).lower().startswith("single positional indexer"):
                        bad_cnt += 1
                        # Impute with global mode, unless globally missing in which case call as 0.0
                        if df[col].isna().all():
                            columns.append(df[col].fillna(0.0, inplace=False))
                        else:
                            columns.append(
                                df[col].fillna(df[col].mode().iloc[0])
                            )
                    else:
                        raise

            data = pd.concat(columns, axis=1)

            if bad_cnt > 0 and not self.validation_mode:
                UserWarning(
                    f"\n{bad_cnt} columns were imputed with the "
                    f"global mode because some of the populations "
                    f"contained only missing data"
                )

            data.drop("pops", axis=1, inplace=True)
        else:
            # Impute global mode.
            imp = SimpleImputer(strategy="most_frequent")

            # replace any columns that are fully missing
            df.loc[:, df.isna().all()] = df.loc[:, df.isna().all()].fillna(0.0)

            data = pd.DataFrame(imp.fit_transform(df))

        if self.iterative_mode:
            data = data.astype(dtype="float32")
        else:
            data = data.astype(dtype="Int8")

        if self.verbose:
            print("Done!")

        if not self.validation_mode:
            return data.values.tolist(), valid_cols
        return data.values, valid_cols

    def write2file(
        self, X: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]
    ) -> None:
        """Write imputed data to file on disk.

        Args:
            X (pandas.DataFrame, numpy.ndarray, List[List[Union[int, float]]]): Imputed data to write to file.

        Raises:
            TypeError: If X is of unsupported type.
        """
        outfile = os.path.join(
            f"{self.prefix}_output",
            "alignments",
            "Unsupervised",
            "ImputeAlleleFreq",
        )

        Path(outfile).mkdir(parents=True, exist_ok=True)

        outfile = os.path.join(outfile, "imputed_012.csv")

        if isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)
        else:
            raise TypeError(
                f"Could not write imputed data because it is of incorrect "
                f"type. Got {type(X)}"
            )

        df.to_csv(outfile, header=False, index=False)


class ImputeMF:
    """Impute missing data using matrix factorization. If ``by_populations=False`` then imputation is by global allele frequency. If ``by_populations=True`` then imputation is by population-wise allele frequency.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData instance.

        latent_features (float, optional): The number of latent variables used to reduce dimensionality of the data. Defaults to 2.

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        missing (int, optional): Missing data value. Defaults to -9.

        prefix (str, optional): Prefix for writing output files. Defaults to "output".

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        **kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool}. "iterative_mode" determines whether ``ImputeAlleleFreq`` is being used as the initial imputer in ``IterativeImputer``.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure",
        >>>    popmapfile="test.popmap",
        >>>)
        >>>
        >>>nmf = ImputeMF(
        >>>    genotype_data=data,
        >>>    by_populations=True,
        >>>)
        >>>
        >>> # Get GenotypeData instance.
        >>>gd_nmf = nmf.imputed

    Raises:
        TypeError: genotype_data cannot be NoneType.
    """

    def __init__(
        self,
        genotype_data,
        *,
        latent_features: int = 2,
        max_iter: int = 100,
        learning_rate: float = 0.0002,
        regularization_param: float = 0.02,
        tol: float = 0.1,
        n_fail: int = 20,
        missing: int = -9,
        prefix: str = "output",
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.max_iter = max_iter
        self.latent_features = latent_features
        self.n_fail = n_fail
        self.learning_rate = learning_rate
        self.tol = tol
        self.regularization_param = regularization_param
        self.missing = missing
        self.prefix = prefix
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)
        self.validation_mode = kwargs.get("validation_mode", False)

        gt = kwargs.get("gt", None)

        if genotype_data is None and gt is None:
            raise TypeError("GenotypeData and gt cannot both be NoneType.")

        if gt is None:
            X = genotype_data.genotypes_012(fmt="numpy")
        else:
            X = gt.copy()
        imputed012 = pd.DataFrame(self.fit_predict(X))
        genotype_data = deepcopy(genotype_data)
        genotype_data.genotypes_012 = imputed012

        if self.validation_mode:
            self.imputed = imputed012.to_numpy()
        else:
            self.imputed = genotype_data

    @property
    def genotypes_012(self):
        return self.imputed.genotypes012

    @property
    def snp_data(self):
        return self.imputed.snp_data

    @property
    def alignment(self):
        return self.imputed.alignment

    def fit_predict(self, X):
        # imputation
        if self.verbose:
            print(f"Doing MF imputation...")
        R = X
        R = R.astype(int)
        R[R == self.missing] = -9
        R = R + 1
        R[R < 0] = 0
        n_row = len(R)
        n_col = len(R[0])
        p = np.random.rand(n_row, self.latent_features)
        q = np.random.rand(n_col, self.latent_features)
        q_t = q.T
        fails = 0
        e_current = None
        for step in range(self.max_iter):
            for i in range(n_row):
                for j in range(n_col):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(p[i, :], q_t[:, j])
                        for k in range(self.latent_features):
                            p[i][k] = p[i][k] + self.learning_rate * (
                                2 * eij * q_t[k][j]
                                - self.regularization_param * p[i][k]
                            )
                            q_t[k][j] = q_t[k][j] + self.learning_rate * (
                                2 * eij * p[i][k]
                                - self.regularization_param * q_t[k][j]
                            )
            e = 0
            for i in range(n_row):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(p[i, :], q_t[:, j]), 2)
                        for k in range(self.latent_features):
                            e = e + (self.regularization_param / 2) * (
                                pow(p[i][k], 2) + pow(q_t[k][j], 2)
                            )
            if e_current is None:
                e_current = e
            else:
                if abs(e_current - e) < self.tol:
                    fails += 1
                else:
                    fails = 0
                e_current = e
            if fails >= self.n_fail:
                break
        nR = np.dot(p, q_t)

        # transform values per-column (i.e., only allowing values found in original)
        tR = self.transform(R, nR)

        # get accuracy of re-constructing non-missing genotypes
        accuracy = self.accuracy(X, tR)

        # insert imputed values for missing genotypes
        fR = X
        fR[X < 0] = tR[X < 0]

        if self.verbose:
            print("Done!")

        return fR

    def transform(self, original, predicted):
        n_row = len(original)
        n_col = len(original[0])
        tR = predicted
        for j in range(n_col):
            observed = predicted[:, j]
            expected = original[:, j]
            options = np.unique(expected[expected != 0])
            for i in range(n_row):
                transform = min(
                    options, key=lambda x: abs(x - predicted[i, j])
                )
                tR[i, j] = transform
        tR = tR - 1
        tR[tR < 0] = -9
        return tR

    def accuracy(self, expected, predicted):
        prop_same = np.sum(expected[expected >= 0] == predicted[expected >= 0])
        tot = expected[expected >= 0].size
        accuracy = prop_same / tot
        return accuracy

    def write2file(
        self, X: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]
    ) -> None:
        """Write imputed data to file on disk.

        Args:
            X (pandas.DataFrame, numpy.ndarray, List[List[Union[int, float]]]): Imputed data to write to file.

        Raises:
            TypeError: If X is of unsupported type.
        """
        outfile = os.path.join(
            f"{self.prefix}_output",
            "alignments",
            "Unsupervised",
            "ImputeMF",
        )

        Path(outfile).mkdir(parents=True, exist_ok=True)

        outfile = os.path.join(outfile, "imputed_012.csv")

        if isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)
        else:
            raise TypeError(
                f"Could not write imputed data because it is of incorrect "
                f"type. Got {type(X)}"
            )

        df.to_csv(outfile, header=False, index=False)


class ImputeRefAllele:
    """Impute missing data by reference allele.

    Args:
        genotype_data (GenotypeData object): GenotypeData instance.

        missing (int, optional): Missing data value. Defaults to -9.

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool, validation_mode: bool, gt: List[List[int]]}. "iterative_mode" determines whether ``ImputeRefAllele`` is being used as the initial imputer in ``IterativeImputer``\. ``gt`` is used internally for the simple imputers during grid searches and validation. If ``genotype_data is None`` then ``gt`` cannot also be None, and vice versa. Only one of ``gt`` or ``genotype_data`` can be set.

    Raises:
        TypeError: genotype_data cannot be NoneType.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    popmapfile="test.popmap",
        >>>)
        >>>
        >>>refallele = ImputeRefAllele(
        >>>     genotype_data=data
        >>>)
        >>>
        >>>gd_refallele = refallele.imputed
    """

    def __init__(
        self,
        genotype_data: GenotypeData,
        *,
        missing: int = -9,
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        if genotype_data is None:
            raise TypeError("GenotypeData instance must be provided.")

        gt = kwargs.get("gt", None)

        if gt is None:
            gt_list = genotype_data.genotypes_012(fmt="list")
        else:
            gt_list = gt

        self.missing = missing
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)
        self.validation_mode = kwargs.get("validation_mode", False)

        # Get reference alleles from GenotypeData object
        self.ref_alleles = genotype_data.ref

        if not self.validation_mode:
            imputed012 = self.fit_predict(gt_list)
            genotype_data = deepcopy(genotype_data)
            genotype_data.genotypes_012 = imputed012
            self.imputed = genotype_data
        else:
            self.imputed = self.fit_predict(gt_list)

    @property
    def genotypes_012(self):
        return self.imputed.genotypes_012

    @property
    def snp_data(self):
        return self.imputed.snp_data

    @property
    def alignment(self):
        return self.imputed.alignment

    def fit_predict(
        self, X: List[List[Union[int, str]]]
    ) -> Union[pd.DataFrame, np.ndarray, List[List[Union[int, str]]]]:
        """Impute missing genotypes using reference alleles.

        Impute using reference alleles. Missing alleles are primarily coded as negative; usually -9.

        Args:
            X (List[List[Union[int, str]]], numpy.ndarray, or pandas.DataFrame): Genotypes obtained from the GenotypeData object.

        Returns:
            pandas.DataFrame, numpy.ndarray, or List[List[Union[int, str]]]: Imputed genotypes of same shape as data.

        Raises:
            TypeError: X must be of type list(list(int or str)), numpy.ndarray,
            or pandas.DataFrame, but got {type(X)}
        """
        if self.verbose:
            print("\nImputing missing data with reference alleles...")

        if isinstance(X, (list, np.ndarray)):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(
                f"X must be of type list(list(int or str)), numpy.ndarray, "
                f"or pandas.DataFrame, but got {type(X)}"
            )

        df = df.astype(df.dtypes)
        df.replace(self.missing, np.nan, inplace=True)

        if df.dtypes[0] == int:
            df.fillna(0, inplace=True)
        else:
            for i, ref in enumerate(self.ref_alleles):
                df[i].fillna(ref, inplace=True)

        if self.verbose:
            print("Done!")

        if not self.validation_mode:
            return df.values.tolist()
        return df.values

    def write2file(
        self, X: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]
    ) -> None:
        """Write imputed data to file on disk.

        Args:
            X (pandas.DataFrame, numpy.ndarray, List[List[Union[int, float]]]): Imputed data to write to file.

        Raises:
            TypeError: If X is of unsupported type.
        """
        outfile = os.path.join(
            f"{self.prefix}_output",
            "alignments",
            "Unsupervised",
            "ImputeRefAllele",
        )

        Path(outfile).mkdir(parents=True, exist_ok=True)

        outfile = os.path.join(outfile, "imputed_012.csv")

        if isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, (np.ndarray, list)):
            df = pd.DataFrame(X)
        else:
            raise TypeError(
                f"Could not write imputed data because it is of incorrect "
                f"type. Got {type(X)}"
            )

        df.to_csv(outfile, header=False, index=False)
