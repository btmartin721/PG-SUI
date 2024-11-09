from snpio import GenotypeEncoder, VCFReader

from pgsui import ImputeVAE


def main():

    gd = VCFReader(
        filename="pgsui/example_data/vcf_files/test.vcf.gz",
        popmapfile="pgsui/example_data/popmaps/test.popmap",
        force_popmap=True,
        verbose=True,
    )

    ge = GenotypeEncoder(gd)

    vae = ImputeVAE(
        gd,
        3,
        tune=True,
        n_trials=100,
        n_jobs=3,
        verbose=1,
        validation_split=0.4,
        seed=42,
    )

    vae_imputed = vae.fit_transform(ge.genotypes_012)
    print(vae_imputed)
    print(vae_imputed.shape)


if __name__ == "__main__":
    main()
