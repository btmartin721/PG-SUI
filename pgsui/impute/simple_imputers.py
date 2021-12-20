import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
import scipy.linalg
import toyplot.pdf
import toyplot as tp
import toytree as tt

# Custom imports
from pgsui.read_input.read_input import GenotypeData
from pgsui.utils import misc
from pgsui.utils.misc import get_processor_name
from pgsui.utils.misc import isnotebook
from pgsui.utils.misc import timer

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    from tqdm import tqdm as progressbar

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        intelex = True
    except ImportError:
        print(
            "Warning: Intel CPU detected but scikit-learn-intelex is not installed. We recommend installing it to speed up computation."
        )
        intelex = False
else:
    intelex = False


class ImputePhylo(GenotypeData):
    """Impute missing data using a phylogenetic tree to inform the imputation.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData object. If not None, some or all of the below options are not required. If None, all the below options are required. Defaults to None.

        alnfile (str or None, optional): Path to PHYLIP or STRUCTURE-formatted file to impute. Defaults to None.

        filetype (str or None, optional): Filetype for the input alignment. Valid options include: "phylip", "structure1row", "structure1rowPopID", "structure2row", "structure2rowPopId". Not required if ``genotype_data`` is defined. Defaults to "phylip".

        popmapfile (str or None, optional): Path to population map file. Required if filetype is "phylip", "structure1row", or "structure2row". If filetype is "structure1rowPopID" or "structure2rowPopID", then the population IDs must be the second column of the STRUCTURE file. Not required if ``genotype_data`` is defined. Defaults to None.

        treefile (str or None, optional): Path to Newick-formatted phylogenetic tree file. Not required if ``genotype_data`` is defined with the ``guidetree`` option. Defaults to None.

        qmatrix_iqtree (str or None, optional): Path to \*.iqtree file containing Rate Matrix Q table. If specified, ``ImputePhylo`` will read the Q matrix from the IQ-TREE output file. Cannot be used in conjunction with ``qmatrix`` argument. Not required if the ``qmatrix`` or ``qmatrix_iqtree`` options were used with the ``GenotypeData`` object. Defaults to None.

        qmatrix (str or None, optional): Path to file containing only a Rate Matrix Q table. Not required if ``genotype_data`` is defined with the qmatrix or qmatrix_iqtree option. Defaults to None.

        str_encodings (Dict[str, int], optional): Integer encodings used in STRUCTURE-formatted file. Should be a dictionary with keys=nucleotides and values=integer encodings. The missing data encoding should also be included. Argument is ignored if using a PHYLIP-formatted file. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}

        prefix (str, optional): Prefix to use with output files.

        save_plots (bool, optional): Whether to save PDF files with genotype imputations for each site to disk. It makes one PDF file per locus, so if you have a lot of loci it will make a lot of PDF files. Defaults to False.

        disable_progressbar (bool, optional): Whether to disable the progress bar during the imputation. Defaults to False.

        kwargs (Dict[str, Any] or None, optional): Additional keyword arguments intended for internal purposes only. Possible arguments: {"column_subset": List[int] or numpy.ndarray[int], "validation_mode": bool}; Subset SNPs by a list of indices. Defauls to None.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>>phylo = ImputePhylo(
        >>>     genotype_data=data,
        >>>     save_plots=True,
        >>>)
        >>>
        >>>phylo_gtdata = phylo.imputed
    """
class ImputePhylo(GenotypeData):
    """Impute missing data using a phylogenetic tree to inform the imputation.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData object. If not None, some or all of the below options are not required. If None, all the below options are required. Defaults to None.

        alnfile (str or None, optional): Path to PHYLIP or STRUCTURE-formatted file to impute. Defaults to None.

        filetype (str or None, optional): Filetype for the input alignment. Valid options include: "phylip", "structure1row", "structure1rowPopID", "structure2row", "structure2rowPopId". Not required if ``genotype_data`` is defined. Defaults to "phylip".

        popmapfile (str or None, optional): Path to population map file. Required if filetype is "phylip", "structure1row", or "structure2row". If filetype is "structure1rowPopID" or "structure2rowPopID", then the population IDs must be the second column of the STRUCTURE file. Not required if ``genotype_data`` is defined. Defaults to None.

        treefile (str or None, optional): Path to Newick-formatted phylogenetic tree file. Not required if ``genotype_data`` is defined with the ``guidetree`` option. Defaults to None.

        siterates (str or None, optional): Path to file containing per-site rates, with 1 rate per line corresponding to 1 site. Not required if ``genotype_data`` is defined with the siterates or siterates_iqtree option. Defaults to None.

        siterates_iqtree (str or None, optional): Path to *.rates file output from IQ-TREE, containing a per-site rate table. If specified, ``ImputePhylo`` will read the site-rates from the IQ-TREE output file. Cannot be used in conjunction with ``siterates`` argument. Not required if the ``siterates`` or ``siterates_iqtree`` options were used with the ``GenotypeData`` object. Defaults to None.

        qmatrix (str or None, optional): Path to file containing only a Rate Matrix Q table. Not required if ``genotype_data`` is defined with the qmatrix or qmatrix_iqtree option. Defaults to None.

        str_encodings (Dict[str, int], optional): Integer encodings used in STRUCTURE-formatted file. Should be a dictionary with keys=nucleotides and values=integer encodings. The missing data encoding should also be included. Argument is ignored if using a PHYLIP-formatted file. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}

        prefix (str, optional): Prefix to use with output files.

        save_plots (bool, optional): Whether to save PDF files with genotype imputations for each site to disk. It makes one PDF file per locus, so if you have a lot of loci it will make a lot of PDF files. Defaults to False.

        write_output (bool, optional): Whether to save the imputed data to disk. Defaults to True.

        disable_progressbar (bool, optional): Whether to disable the progress bar during the imputation. Defaults to False.

        kwargs (Dict[str, Any] or None, optional): Additional keyword arguments intended for internal purposes only. Possible arguments: {"column_subset": List[int] or numpy.ndarray[int]}; Subset SNPs by a list of indices. Defauls to None.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>>phylo = ImputePhylo(
        >>>     genotype_data=data,
        >>>     save_plots=True,
        >>>)
        >>>
        >>>phylo_gtdata = phylo.imputed
    """
    def __init__(
        self,
        *,
        genotype_data: Optional[Any] = None,
        alnfile: Optional[str] = None,
        filetype: Optional[str] = None,
        popmapfile: Optional[str] = None,
        treefile: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        siterates_iqtree: Optional[str] = None,
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
        **kwargs: Optional[Any],
    ) -> None:
        super().__init__()

        self.alnfile = alnfile
        self.filetype = filetype
        self.popmapfile = popmapfile
        self.treefile = treefile
        self.qmatrix_iqtree = qmatrix_iqtree
        self.qmatrix = qmatrix
        self.siterates = siterates
        self.siterates_iqtree = siterates_iqtree
        self.str_encodings = str_encodings
        self.prefix = prefix
        self.save_plots = save_plots
        self.disable_progressbar = disable_progressbar
        self.column_subset = kwargs.get("column_subset", None)
        self.validation_mode = kwargs.get("validation_mode", False)

        self.valid_sites = None
        self.valid_sites_count = None

        self.validate_arguments(genotype_data)
        data, tree, q, site_rates = self.parse_arguments(genotype_data)

        if self.validation_mode == True:
            imputed012 = self.impute_phylo(tree, data, q, site_rates)

            imputed_filename = genotype_data.decode_imputed(
                imputed012, write_output=True, prefix=prefix
            )

            ft = genotype_data.filetype

            if ft.lower().startswith("structure") and ft.lower().endswith(
                "row"
            ):
                ft += "PopID"

            self.imputed = GenotypeData(
                filename=imputed_filename,
                filetype=ft,
                guidetree=genotype_data.guidetree,
                qmatrix_iqtree=genotype_data.qmatrix_iqtree,
                qmatrix=genotype_data.qmatrix,
                siterates=genotype_data.siterates,
                siterates_iqtree=genotype_data.siterates_iqtre,
                verbose=False,
            )

        else:
            self.imputed = self.impute_phylo(tree, data, q, site_rates)

    def nbiallelic(self) -> int:
        """Get the number of remaining bi-allelic sites after imputation.

        Returns:
            int: Number of bi-allelic sites remaining after imputation.
        """
        return len(self.imputed.columns)

    def parse_arguments(
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
        if genotype_data is not None:
            data = genotype_data.snpsdict
            self.filetype = genotype_data.filetype

        elif self.alnfile is not None:
            self.parse_filetype(self.filetype, self.popmapfile)

        if genotype_data.tree is not None and self.treefile is None:
            tree = genotype_data.tree

        elif genotype_data.tree is not None and self.treefile is not None:
            print(
                "WARNING: Both genotype_data.tree and treefile are defined; using local definition"
            )
            tree = self.read_tree(self.treefile)

        elif genotype_data.tree is None and self.treefile is not None:
            tree = self.read_tree(self.treefile)

        #read (optional) Q-matrix
        if (
            genotype_data.q is not None
            and self.qmatrix is None
            and self.qmatrix_iqtree is None
        ):
            q = genotype_data.q

        elif genotype_data.q is None:
            if self.qmatrix is not None:
                q = self.q_from_file(self.qmatrix)
            elif self.qmatrix_iqtree is not None:
                q = self.q_from_iqtree(self.qmatrix_iqtree)

        elif genotype_data.q is not None:
            if self.qmatrix is not None:
                print(
                    "WARNING: Both genotype_data.q and qmatrix are defined; "
                    "using local definition"
                )
                q = self.q_from_file(self.qmatrix)
            if self.qmatrix_iqtree is not None:
                print(
                    "WARNING: Both genotype_data.q and qmatrix are defined; "
                    "using local definition"
                )
                q = self.q_from_iqtree(self.qmatrix_iqtree)

        #read (optional) site-specific substitution rates
        site_rates = None
        if (
            genotype_data.site_rates is not None
            and self.siterates is None
            and self.siterates_iqtree is None
        ):
            site_rates = genotype_data.site_rates
        elif genotype_data.site_rates is None:
            if self.siterates is not None:
                site_rates = self.siterates_from_file(self.siterates)
            elif self.siterates_iqtree is not None:
                site_rates = self.siterates_from_iqtree(self.siterates_iqtree)

        elif genotype_data.site_rates is not None:
            if self.siterates is not None:
                print(
                    "WARNING: Both genotype_data.site_rates and siterates are defined; "
                    "using local definition"
                )
                site_rates = self.siterates_from_file(self.siterates)
            if self.siterates_iqtree is not None:
                print(
                    "WARNING: Both genotype_data.site_rates and siterates are defined; "
                    "using local definition"
                )
                site_rates = self.siterates_from_iqtree(self.siterates_iqtree)
        return (data, tree, q, site_rates)

    def validate_arguments(self, genotype_data: Any) -> None:
        """Validate that the correct arguments were supplied.

        Args:
            genotype_data (GenotypeData object): Input GenotypeData object.

        Raises:
            TypeError: Cannot define both genotype_data and alnfile.
            TypeError: Must define either genotype_data or phylipfile.
            TypeError: Must define either genotype_data.tree or treefile.
            TypeError: filetype must be defined if genotype_data is None.
            TypeError: Q rate matrix must be defined.
            TypeError: qmatrix and qmatrix_iqtree cannot both be defined.
        """
        if genotype_data is not None and self.alnfile is not None:
            raise TypeError("genotype_data and alnfile cannot both be defined")

        if genotype_data is None and self.alnfile is None:
            raise TypeError("Either genotype_data or phylipfle must be defined")

        if genotype_data.tree is None and self.treefile is None:
            raise TypeError(
                "Either genotype_data.tree or treefile must be defined"
            )

        if genotype_data is None and self.filetype is None:
            raise TypeError("filetype must be defined if genotype_data is None")

        if (
            genotype_data is None
            and self.qmatrix is None
            and self.qmatrix_iqtree is None
        ):
            raise TypeError(
                "q matrix must be defined in either genotype_data, "
                "qmatrix_iqtree, or qmatrix"
            )

        if self.qmatrix is not None and self.qmatrix_iqtree is not None:
            raise TypeError("qmatrix and qmatrix_iqtree cannot both be defined")

    def print_q(self, q: pd.DataFrame) -> None:
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

    def is_int(self, val: Union[str, int]) -> bool:
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

    def impute_phylo(
        self,
        tree: tt.tree,
        genotypes: Dict[str, List[Union[str, int]]],
        Q: pd.DataFrame,
        site_rates = None,
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

        Returns:
            pandas.DataFrame: Imputed genotypes.

        Raises:
            IndexError: If index does not exist when trying to read genotypes.
            AssertionError: Sites must have same lengths.
            AssertionError: Missing data still found after imputation.
        """
        try:
            if list(genotypes.values())[0][0][1] == "/":
                genotypes = self.str2iupac(genotypes, self.str_encodings)
        except IndexError:
            if self.is_int(list(genotypes.values())[0][0][0]):
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
            node_lik = dict()

            # LATER: Need to get site rates
            rate = 1.0
            if site_rates is not None:
                rate = site_rates[snp_index]

            site_Q = Q.copy(deep=True) * rate
            #print(site_Q)

            # calculate state likelihoods for internal nodes
            for node in tree.treenode.traverse("postorder"):
                if node.is_leaf():
                    continue

                if node.idx not in node_lik:
                    node_lik[node.idx] = None

                for child in node.get_leaves():
                    # get branch length to child
                    # bl = child.edge.length
                    # get transition probs
                    pt = self.transition_probs(site_Q, child.dist)
                    if child.is_leaf():
                        if child.name in genotypes:
                            # get genotype
                            sum = None

                            for allele in self.get_iupac_full(
                                genotypes[child.name][snp_index]
                            ):
                                if sum is None:
                                    sum = list(pt[allele])
                                else:
                                    sum = [
                                        sum[i] + val
                                        for i, val in enumerate(
                                            list(pt[allele])
                                        )
                                    ]

                            if node_lik[node.idx] is None:
                                node_lik[node.idx] = sum

                            else:
                                node_lik[node.idx] = [
                                    sum[i] * val
                                    for i, val in enumerate(node_lik[node.idx])
                                ]
                        else:
                            # raise error
                            sys.exit(
                                f"Error: Taxon {child.name} not found in "
                                f"genotypes"
                            )

                    else:
                        l = self.get_internal_lik(pt, node_lik[child.idx])
                        if node_lik[node.idx] is None:
                            node_lik[node.idx] = l

                        else:
                            node_lik[node.idx] = [
                                l[i] * val
                                for i, val in enumerate(node_lik[node.idx])
                            ]

            # infer most likely states for tips with missing data
            # for each child node:
            bads = list()
            for samp in genotypes.keys():
                if genotypes[samp][snp_index].upper() == "N":
                    bads.append(samp)
                    # go backwards into tree until a node informed by
                    # actual data
                    # is found
                    # node = tree.search_nodes(name=samp)[0]
                    node = tree.idx_dict[
                        tree.get_mrca_idx_from_tip_labels(names=samp)
                    ]
                    dist = node.dist
                    node = node.up
                    imputed = None

                    while node and imputed is None:
                        if self.allMissing(
                            tree, node.idx, snp_index, genotypes
                        ):
                            dist += node.dist
                            node = node.up

                        else:
                            pt = self.transition_probs(site_Q, dist)
                            lik = self.get_internal_lik(pt, node_lik[node.idx])
                            maxpos = lik.index(max(lik))
                            if maxpos == 0:
                                imputed = "A"

                            elif maxpos == 1:
                                imputed = "C"

                            elif maxpos == 2:
                                imputed = "G"

                            else:
                                imputed = "T"

                    genotypes[samp][snp_index] = imputed

            if self.save_plots:
                self.draw_imputed_position(
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

        imp_snps, self.valid_sites, self.valid_sites_count = self.convert_012(
            df.to_numpy().tolist(), impute_mode=True
        )

        df_imp = pd.DataFrame.from_records(imp_snps)

        return df_imp

    def get_nuc_colors(self, nucs: List[str]) -> List[str]:
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

    def label_bads(
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

    def draw_imputed_position(
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

        labels = self.label_bads(tree.get_tip_labels(), labels, bads)

        colors = self.get_nuc_colors(colors)

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

    def allMissing(
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

    def get_internal_lik(
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
            sum = 0.0
            for v in col:
                sum += v * val
            ret.append(sum)
        return ret

    def transition_probs(self, Q: pd.DataFrame, t: float) -> pd.DataFrame:
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

    def get_iupac_full(self, char: str) -> List[str]:
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


class ImputeAlleleFreq(GenotypeData):
    """Impute missing data by global allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations.

    Args:
        genotype_data (GenotypeData object or None): GenotypeData instance. Required keyword argument. Defaults to None.

        by_populations (bool, optional): Whether or not to impute by population or globally. Defaults to False (global allele frequency).

        pops (List[Union[str, int]] or None, optional): Population IDs in the same order as the samples. If ``by_populations=True``\, then either ``pops`` or ``genotype_data`` must be defined. If both are defined, the ``pops`` argument will take priority. Defaults to None.

        diploid (bool, optional): When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency. Defaults to True.

        default (int, optional): Value to set if no alleles sampled at a locus. Defaults to 0.

        missing (int, optional): Missing data value. Defaults to -9.

        prefix (str, optional): Prefix for writing output files. Defaults to "output".

        output_format (str, optional): Format of output imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list. Defaults to "df".

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool, validation_mode: bool, gt: List[List[int]]}. "iterative_mode" determines whether ``ImputeAlleleFreq`` is being used as the initial imputer in ``IterativeImputer``\. ``gt`` is used internally for the simple imputers during grid searches and validation. If ``genotype_data is None`` then ``gt`` cannot also be None, and vice versa. Only one of ``gt`` or ``genotype_data`` can be set.

    Raises:
        TypeError: genotype_data and gt cannot both be NoneType.
        TypeError: genotype_data and gt cannot both be provided.
        TypeError: Either pops or genotype_data must be defined if by_populations is True.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID",
        >>>    guidetree="test.tre",
        >>>    qmatrix_iqtree="test.iqtree"
        >>>)
        >>>
        >>>afpop = ImputeAlleleFreq(
        >>>     genotype_data=data,
        >>>     by_populations=True,
        >>>)
        >>>
        >>>afpop_gtdata = afpop.imputed
    """

    def __init__(
        self,
        *,
        genotype_data: Optional[Any] = None,
        by_populations: bool = False,
        pops: Optional[List[Union[str, int]]] = None,
        diploid: bool = True,
        default: int = 0,
        missing: int = -9,
        prefix: str = "output",
        output_format: str = "df",
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:

        super().__init__()

        gt = kwargs.get("gt", None)

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            gt_list = genotype_data.genotypes_list
        elif gt is not None:
            gt_list = gt

        if by_populations:
            if pops is None and genotype_data is None:
                raise TypeError(
                    "When by_populations is True, either pops or genotype_data must be defined"
                )

            if genotype_data is not None and pops is not None:
                print(
                    "WARNING: Both pops and genotype_data are defined. Using populations from pops argument"
                )
                self.pops = pops

            elif genotype_data is not None and pops is None:
                self.pops = genotype_data.populations

            elif genotype_data is None and pops is not None:
                self.pops = pops

        else:
            if pops is not None:
                print(
                    "WARNING: by_populations is False but pops is defined. Setting pops to None"
                )

            self.pops = None

        self.diploid = diploid
        self.default = default
        self.missing = missing
        self.prefix = prefix
        self.output_format = output_format
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)
        self.validation_mode = kwargs.get("validation_mode", False)

        if self.validation_mode == True:
            imputed012, self.valid_cols = self.fit_predict(gt_list)

            imputed_filename = genotype_data.decode_imputed(
                imputed012, write_output=True, prefix=prefix
            )

            ft = genotype_data.filetype

            if ft.lower().startswith("structure") and ft.lower().endswith(
                "row"
            ):
                ft += "PopID"

            self.imputed = GenotypeData(
                filename=imputed_filename,
                filetype=ft,
                guidetree=genotype_data.guidetree,
                qmatrix_iqtree=genotype_data.qmatrix_iqtree,
                qmatrix=genotype_data.qmatrix,
                siterates=genotype_data.siterates,
                siterates_iqtree=genotype_data.siterates_iqtree,
                verbose=False,
            )

        else:
            self.imputed, self.valid_cols = self.fit_predict(gt_list)

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
            TypeError: X must be either 2D list, numpy.ndarray, or pandas.DataFrame.

            ValueError: Unknown output_format type specified.
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

        df.replace(self.missing, np.nan, inplace=True)

        data = pd.DataFrame()
        valid_cols = list()
        bad_cnt = 0
        if self.pops is not None:
            # Impute per-population mode.
            # Loop method is faster (by 2X) than no-loop transform.
            df["pops"] = self.pops
            for col in df.columns:
                try:
                    data[col] = df.groupby(["pops"], sort=False)[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0])
                    )

                    # If all populations contained at least one non-NaN value.
                    if col != "pops":
                        valid_cols.append(col)

                except IndexError as e:
                    if str(e).lower().startswith("single positional indexer"):
                        bad_cnt += 1
                        # Impute with global mode
                        data[col] = df[col].fillna(df[col].mode().iloc[0])
                    else:
                        raise

            if bad_cnt > 0:
                print(
                    f"Warning: {bad_cnt} columns were imputed with the global "
                    f"mode because some of the populations "
                    f"contained only missing data"
                )

            data.drop("pops", axis=1, inplace=True)
        else:
            # Impute global mode.
            # No-loop method was faster for global.
            data = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=1)

        if self.iterative_mode:
            data = data.astype(dtype="float32")
        else:
            data = data.astype(dtype="Int8")

        if self.verbose:
            print("Done!")

        if self.output_format == "df":
            return data, valid_cols

        elif self.output_format == "array":
            return data.to_numpy(), valid_cols

        elif self.output_format == "list":
            return data.values.tolist(), valid_cols

        else:
            raise ValueError("Unknown output_format type specified!")

    def write2file(
        self, X: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]
    ) -> None:
        """Write imputed data to file on disk.

        Args:
            X (pandas.DataFrame, numpy.ndarray, List[List[Union[int, float]]]): Imputed data to write to file.

        Raises:
            TypeError: If X is of unsupported type.
        """
        outfile = f"{self.prefix}_imputed_012.csv"

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

class ImputeNMF(GenotypeData):
    """Impute missing data using matrix factorization. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations.

    Args:
        genotype_data (GenotypeData object or None, optional): GenotypeData instance. If ``genotype_data`` is not defined, then ``gt`` must be defined instead, and they cannot both be defined. Defaults to None.

        gt (List[int] or None, optional): List of 012-encoded genotypes to be imputed. Either ``gt`` or ``genotype_data`` must be defined, and they cannot both be defined. Defaults to None.

        latent_features (float, optional): The number of latent variables used to reduce dimensionality of the data. Defaults to 2.

        learning_rate (float, optional): The learning rate for the optimizers. Adjust if the loss is learning too slowly. Defaults to 0.1.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        missing (int, optional): Missing data value. Defaults to -9.

        prefix (str, optional): Prefix for writing output files. Defaults to "output".

        write_output (bool, optional): Whether to save imputed output to a file. If ``write_output`` is False, then just returns the imputed values as a pandas.DataFrame object. If ``write_output`` is True, then it saves the imputed data as a CSV file called ``<prefix>_imputed_012.csv``.

        output_format (str, optional): Format of output imputed matrix. Possible values include: "df" for a pandas.DataFrame object, "array" for a numpy.ndarray object, and "list" for a 2D list. Defaults to "df".

        verbose (bool, optional): Whether to print status updates. Set to False for no status updates. Defaults to True.

        **kwargs (Dict[str, Any]): Additional keyword arguments to supply. Primarily for internal purposes. Options include: {"iterative_mode": bool}. "iterative_mode" determines whether ``ImputeAlleleFreq`` is being used as the initial imputer in ``IterativeImputer``.

    Attributes:
        imputed (GenotypeData): New GenotypeData instance with imputed data.

    Example:
        >>>data = GenotypeData(
        >>>    filename="test.str",
        >>>    filetype="structure2rowPopID"
        >>>)
        >>>
        >>>nmf = ImputeNMF(
        >>>     genotype_data=data,
        >>>)
        >>>
        >>>nmf_gtdata = nmf.imputed

    Raises:
        TypeError: genotype_data and gt cannot both be NoneType.
        TypeError: genotype_data and gt cannot both be provided.
        TypeError: Either pops or genotype_data must be defined if by_populations is True.
    """

    def __init__(
        self,
        *,
        genotype_data=None,
        gt=None,
        latent_features: int = 2,
        max_iter: int = 100,
        learning_rate: float = 0.0002,
        regularization_param: float = 0.02,
        tol: float = 0.1,
        n_fail: int = 20,
        missing: int = -9,
        prefix: str = "output",
        write_output: bool = True,
        output_format= None,
        verbose: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:

        super().__init__()

        self.max_iter = max_iter
        self.latent_features = latent_features
        self.n_fail = n_fail
        self.learning_rate = learning_rate
        self.tol = tol
        self.regularization_param = regularization_param
        self.missing = missing
        self.prefix = prefix
        self.output_format = output_format
        self.verbose = verbose
        self.iterative_mode = kwargs.get("iterative_mode", False)
        self.validation_mode = kwargs.get("validation_mode", False)

        if genotype_data is None and gt is None:
            raise TypeError("genotype_data and gt cannot both be NoneType")

        if genotype_data is not None and gt is not None:
            raise TypeError("genotype_data and gt cannot both be used")

        if genotype_data is not None:
            X = genotype_data.genotypes_nparray
        elif gt is not None:
            X = gt

        if self.validation_mode == True:
            imputed012 = pd.DataFrame(self.fit_predict(X))

            imputed_filename = genotype_data.decode_imputed(
                imputed012, write_output=True, prefix=prefix
            )

            ft = genotype_data.filetype

            if ft.lower().startswith("structure") and ft.lower().endswith(
                "row"
            ):
                ft += "PopID"

            self.imputed = GenotypeData(
                filename=imputed_filename,
                filetype=ft,
                guidetree=genotype_data.guidetree,
                qmatrix_iqtree=genotype_data.qmatrix_iqtree,
                qmatrix=genotype_data.qmatrix,
                siterates=genotype_data.siterates,
                siterates_iqtree=genotype_data.siterates_iqtree,
                verbose=False,
            )
        else:
            self.imputed = pd.DataFrame(self.fit_predict(X))
            if self.output_format is not None:
                if self.output_format == "df":
                    pass
                elif self.output_format == "array":
                    self.imputed = nX
                elif self.output_format == "list":
                    self.imputed = self.imputed.tolist()

            if write_output:
                self.write2file(self.imputed)


    def fit_predict(self, X):
        #imputation
        if self.verbose:
            print(f"Doing NMF imputation without grid search...")
        R = X
        R[R==self.missing]=-9
        R = R+1
        R[R<0] = 0
        n_row = len(R)
        n_col = len(R[0])
        p = np.random.rand(n_row,self.latent_features)
        q = np.random.rand(n_col,self.latent_features)
        q_t = q.T
        fails=0
        e_current = None
        for step in range(self.max_iter):
            for i in range(n_row):
                for j in range(n_col):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(p[i,:],q_t[:,j])
                        for k in range(self.latent_features):
                            p[i][k] = p[i][k] + self.learning_rate * (2 * eij * q_t[k][j] - self.regularization_param * p[i][k])
                            q_t[k][j] = q_t[k][j] + self.learning_rate * (2 * eij * p[i][k] - self.regularization_param * q_t[k][j])
            e = 0
            for i in range(n_row):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(p[i,:],q_t[:,j]), 2)
                        for k in range(self.latent_features):
                            e = e + (self.regularization_param/2) * ( pow(p[i][k],2) + pow(q_t[k][j],2) )
            if e_current is None:
                e_current = e
            else:
                if abs(e_current - e) < self.tol:
                    fails+=1
                else:
                    fails=0
                e_current=e
            if fails >= self.n_fail:
                break
        nR = np.dot(p, q_t)

        #transform values per-column (i.e., only allowing values found in original)
        tR=self.transform(R, nR)

        #get accuracy of re-constructing non-missing genotypes
        accuracy=self.accuracy(X, tR)

        #insert imputed values for missing genotypes
        fR=X
        fR[X<0] = tR[X<0]

        if self.verbose:
            print("Done!")

        return(fR)

    def transform(self, original, predicted):
        n_row = len(original)
        n_col = len(original[0])
        tR=predicted
        for j in range(n_col):
            observed = predicted[:,j]
            expected = original[:,j]
            options = np.unique(expected[expected != 0])
            for i in range(n_row):
                transform = min(options, key=lambda x:abs(x-predicted[i,j]))
                tR[i,j]=transform
        tR = tR-1
        tR[tR<0] = -9
        return(tR)

    def accuracy(self, expected, predicted):
        prop_same=np.sum(expected[expected>=0]==predicted[expected>=0])
        tot=expected[expected>=0].size
        accuracy=prop_same/tot
        return(accuracy)

    def write2file(
        self, X: Union[pd.DataFrame, np.ndarray, List[List[Union[int, float]]]]
    ) -> None:
        """Write imputed data to file on disk.

        Args:
            X (pandas.DataFrame, numpy.ndarray, List[List[Union[int, float]]]): Imputed data to write to file.

        Raises:
            TypeError: If X is of unsupported type.
        """
        outfile = f"{self.prefix}_imputed_012.csv"

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
