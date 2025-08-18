import importlib.resources as pkg_resources
import pprint

from snpio import GenotypeEncoder, VCFReader, NRemover2

from pgsui import (
    ImputeAutoencoder,
    ImputeHistGradientBoosting,
    ImputeNLPCA,
    ImputeRandomForest,
    ImputeUBP,
    ImputeVAE,
)


def main():
    # Locate the correct installed path for VCF and popmap files
    vcf_path = (
        pkg_resources.files("pgsui")
        / "example_data/vcf_files/phylogen_subset14K.vcf.gz"
    )
    popmap_path = (
        pkg_resources.files("pgsui") / "example_data/popmaps/phylogen_nomx.popmap"
    )

    print(f"Using VCF file: {vcf_path}")
    print(f"Using popmap file: {popmap_path}")

    # Pass absolute paths to VCFReader
    gd = VCFReader(
        filename=vcf_path,
        popmapfile=popmap_path,
        force_popmap=True,
        verbose=True,
        include_pops=["EA", "GU", "TT", "ON"],
    )

    nrm = NRemover2(gd)

    gd_filt = (
        nrm.filter_missing_sample(0.8)
        .filter_missing(0.5)
        .filter_missing_pop(0.5)
        .filter_monomorphic(exclude_heterozygous=True)
        .filter_singletons(exclude_heterozygous=True)
        .filter_biallelic(exclude_heterozygous=True)
        .filter_mac(3)
        .resolve()
    )

    # ubp = ImputeUBP(
    #     gd_filt,
    #     n_jobs=8,
    #     verbose=1,
    #     sim_prop_missing=0.5,
    #     sim_strategy="random_inv_multinom",
    #     model_device="cpu",
    #     model_dropout_rate=0.45,
    #     model_hidden_activation="elu",
    #     model_latent_dim=25,
    #     model_learning_rate=0.001,
    #     model_lr_input_factor=2.0,
    #     model_gamma=1.0,
    #     model_num_hidden_layers=4,
    #     tune=True,
    #     prefix="pgsui_ubp_test_noconv_2",
    #     tune_n_trials=100,
    #     model_batch_size=64,
    #     weights_temperature=1.0,
    #     weights_alpha=1.0,
    #     weights_log_scale=False,
    #     model_use_convolution=False,  # Set to True to use convolutional UBP
    # )

    nlpca = ImputeNLPCA(
        gd_filt,
        n_jobs=1,
        verbose=1,
        sim_prop_missing=0.1,
        sim_strategy="random_balanced_multinom",
        model_device="cpu",
        tune=True,
        prefix="pgsui_nlpca_test",
        tune_n_trials=500,
        model_batch_size=64,
    )

    ge = GenotypeEncoder(gd_filt)
    X = ge.genotypes_012

    # ubp.fit_transform(X)
    nlpca.fit_transform(X)


if __name__ == "__main__":
    main()
