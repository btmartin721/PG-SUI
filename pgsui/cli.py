import importlib.resources as pkg_resources
import pprint

import numpy as np
from snpio import GenotypeEncoder, VCFReader

from pgsui import ImputeUBP, ImputeVAE


def main():
    # Locate the correct installed path for VCF and popmap files
    vcf_path = (
        pkg_resources.files("pgsui")
        / "example_data/vcf_files/phylogen_subset14K.vcf_sorted.vcf.gz"
    )
    popmap_path = (
        pkg_resources.files("pgsui") / "example_data/popmaps/phylogen_nomx.popmap"
    )

    print(f"Using VCF file: {vcf_path}")
    print(f"Using popmap file: {popmap_path}")

    # Pass absolute paths to VCFReader
    gd = VCFReader(
        filename=str(vcf_path),
        popmapfile=str(popmap_path),
        force_popmap=True,
        verbose=True,
    )

    snp_data = gd.snp_data

    gd.loci_indices = np.ones((len(snp_data[0])), dtype=bool)
    gd.sample_indices = np.ones((len(snp_data)), dtype=bool)

    ge = GenotypeEncoder(gd)

    vae = ImputeVAE(
        gd,
        weights_temperature=3.0,
        weights_normalize=True,
        weights_alpha=0.1,
        weights_log_scale=False,
        tune=False,
        tune_metric="pr_macro",
        tune_n_trials=100,
        model_early_stop_gen=20,
        model_min_epochs=20,
        model_validation_split=0.21,
        model_learning_rate=0.0001,
        model_latent_dim=2,
        model_hidden_activation="elu",
        model_l1_penalty=0.0,
        model_lr_patience=8,
        model_device="cpu",
        n_jobs=8,
        verbose=1,
        seed=42,
        model_gamma=2.0,
        model_batch_size=16,
        sim_strategy="random_balanced_inv_global",
        sim_prop_missing=0.3,
    )

    vae_imputed = vae.fit_transform(ge.genotypes_012)
    print(vae_imputed)
    print(vae_imputed.shape)
    print("Best found parameters:")
    pprint.pprint(vae.best_params_, indent=4)

    print("Scoring metrics:")
    pprint.pprint(vae.metrics_, indent=4)


if __name__ == "__main__":
    main()
