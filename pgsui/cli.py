import importlib.resources as pkg_resources

from snpio import VCFReader

from pgsui import (
    ImputeAutoencoder,
    ImputeMostFrequent,
    ImputeNLPCA,
    ImputeRefAllele,
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

    ubp = ImputeUBP(
        gd,
        prefix="pgsui_test",
        n_jobs=8,
        tune=False,
        tune_n_trials=100,
        model_batch_size=64,
        verbose=True,
    )

    ubp.fit()
    X_imputed = ubp.transform()
    print(X_imputed)
    print(X_imputed.shape)

    vae = ImputeVAE(
        gd,
        n_jobs=1,
        verbose=1,
        model_device="cpu",
        tune=False,
        prefix="pgsui_test",
        tune_n_trials=50,
        model_batch_size=64,
        model_learning_rate=0.0008020404122071253,
        model_latent_dim=28,
        model_l1_penalty=1.682421969883547e-06,
        model_gamma=4.5,
        model_dropout_rate=0.01,
        weights_beta=0.99,
        weights_max_ratio=2.0,
        model_hidden_activation="leaky_relu",
        model_early_stop_gen=25,
        model_num_hidden_layers=13,
        plot_format="png",
    )

    vae.fit()
    X_imputed = vae.transform()
    print(X_imputed)
    print(X_imputed.shape)

    ae = ImputeAutoencoder(
        gd,
        n_jobs=1,
        verbose=1,
        model_device="cpu",
        tune=False,
        prefix="pgsui_test",
        tune_n_trials=10,
        model_batch_size=64,
        tune_metric="f1",
        plot_format="png",
    )

    ae.fit()
    X_imputed = ae.transform()
    print(X_imputed)
    print(X_imputed.shape)

    nlpca = ImputeNLPCA(
        gd,
        verbose=1,
        model_device="cpu",
        tune=False,
        prefix="pgsui_test",
        tune_n_trials=20,
        model_batch_size=64,
    )

    nlpca.fit()
    X_imputed = nlpca.transform()
    print(X_imputed)
    print(X_imputed.shape)

    mode = ImputeMostFrequent(gd, prefix="pgsui_test", verbose=True, seed=42)
    mode.fit()
    X_imputed = mode.transform()
    print(X_imputed)
    print(X_imputed.shape)

    mode.fit()
    X_imputed = mode.transform()
    print(X_imputed)
    print(X_imputed.shape)

    ref = ImputeRefAllele(gd, prefix="pgsui_test", verbose=True, seed=42)
    ref.fit()
    X_imputed = ref.transform()
    print(X_imputed)
    print(X_imputed.shape)


if __name__ == "__main__":
    main()
