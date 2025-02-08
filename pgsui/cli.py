import importlib.resources as pkg_resources
import pprint

from snpio import GenotypeEncoder, VCFReader

from pgsui import ImputeAutoencoder, ImputeNLPCA, ImputeUBP, ImputeVAE


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

    ge = GenotypeEncoder(gd)

    # ubp = ImputeUBP(
    #     gd,
    #     weights_temperature=1.0,
    #     weights_normalize=False,
    #     weights_alpha=2.0,
    #     weights_log_scale=True,
    #     tune=True,
    #     tune_metric="pr_macro",
    #     tune_n_trials=250,
    #     model_early_stop_gen=20,
    #     model_min_epochs=20,
    #     model_validation_split=0.21,
    #     model_learning_rate=0.0008886973483648648,
    #     model_latent_dim=4,
    #     model_hidden_activation="leaky_relu",
    #     model_l1_penalty=0.0,
    #     model_lr_patience=8,
    #     model_dropout_rate=0.0,
    #     model_device="cpu",
    #     model_num_hidden_layers=8,
    #     model_batch_size=32,
    #     model_epochs=100,
    #     model_gamma=2.0,
    #     n_jobs=8,
    #     verbose=1,
    #     seed=42,
    #     sim_strategy="random_inv_multinom,
    #     sim_prop_missing=0.3,
    # )

    # nlpca = ImputeNLPCA(
    #     gd,
    #     weights_temperature=1.0,
    #     weights_normalize=False,
    #     weights_alpha=2.0,
    #     weights_log_scale=True,
    #     tune=False,
    #     tune_metric="pr_macro",
    #     tune_n_trials=50,
    #     model_early_stop_gen=20,
    #     model_min_epochs=20,
    #     model_validation_split=0.21,
    #     model_learning_rate=0.00027532421267350253,
    #     model_latent_dim=4,
    #     model_hidden_activation="leaky_relu",
    #     model_l1_penalty=0.0,
    #     model_lr_patience=8,
    #     model_dropout_rate=0.25,
    #     model_device="cpu",
    #     model_num_hidden_layers=1,
    #     model_batch_size=32,
    #     model_epochs=100,
    #     model_gamma=2.0,
    #     n_jobs=8,
    #     verbose=1,
    #     seed=42,
    #     sim_strategy="random_inv_multinom",
    #     sim_prop_missing=0.3,
    # )

    vae = ImputeAutoencoder(
        gd,
        weights_temperature=1.0,
        weights_normalize=False,
        weights_alpha=2.0,
        weights_log_scale=True,
        tune=False,
        tune_metric="pr_macro",
        tune_n_trials=50,
        model_early_stop_gen=20,
        model_min_epochs=20,
        model_learning_rate=0.00027532421267350253,
        model_latent_dim=4,
        model_hidden_activation="leaky_relu",
        model_l1_penalty=0.0,
        model_lr_patience=8,
        model_dropout_rate=0.25,
        model_device="cpu",
        model_num_hidden_layers=1,
        model_batch_size=32,
        model_epochs=100,
        model_gamma=2.0,
        n_jobs=8,
        verbose=1,
        seed=42,
        sim_strategy="random_inv_multinom",
        sim_prop_missing=0.3,
    )

    imputed = vae.fit_transform(ge.genotypes_012)
    print(imputed)
    print(imputed.shape)
    print("Best found parameters:")
    pprint.pprint(vae.best_params_, indent=4)

    print("Scoring metrics:")
    pprint.pprint(vae.metrics_, indent=4)


if __name__ == "__main__":
    main()
