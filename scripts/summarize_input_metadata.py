import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl_settings = {
    "figure.dpi": 300,
    "axes.titlesize": "x-large",
    "axes.labelsize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "legend.fontsize": "x-large",
}
mpl.rcParams.update(mpl_settings)


def main():
    # Load metadata
    df = pd.read_csv("subset_metadata.csv", header=0)
    df["prefix"] = df["filename"].str.replace(".phy", "", regex=False)
    df = df.set_index("prefix")
    df = df.drop(columns=["filename"])
    df = df.rename(columns={"n_samples": "Sample Count", "n_sites": "Site Count"})
    df["Sample Count"] = df["Sample Count"].astype(int)
    df["Site Count"] = df["Site Count"].astype(int)

    desc = df.describe().T
    desc.to_csv("subset_metadata_summary.csv")

    # Summarize metadata
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
    ax = axes[0]
    ax = sns.histplot(
        data=df, x="Sample Count", bins=20, kde=False, color="darkorchid", ax=ax
    )
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Frequency")
    ax.set_title("Samples per Dataset")

    ax = axes[1]
    ax = sns.histplot(
        data=df, x="Site Count", bins=20, kde=False, color="cornflowerblue", ax=ax
    )
    ax.set_xlabel("Number of Sites")
    ax.set_ylabel("Frequency")
    ax.set_title("Sites per Dataset")
    fig.tight_layout()
    fig.savefig("subset_distributions.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
