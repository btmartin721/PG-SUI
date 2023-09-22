import msprime
import numpy as np
import demes
import math


def add_monomorphic_sites(ts, ancestral_states):
    """
    Add monomorphic sites to a tree sequence.

    Args:
        ts (TreeSequence): The original tree sequence.
        ancestral_states (list): List of ancestral states to use for the monomorphic sites.

    Returns:
        TreeSequence: A new tree sequence with monomorphic sites added.
    """
    tables = ts.dump_tables()
    existing_positions = set(site.position for site in ts.sites())
    sequence_length = int(ts.sequence_length)

    for pos in range(sequence_length):
        if pos not in existing_positions:
            ancestral_state = ancestral_states[pos % len(ancestral_states)]
            tables.sites.add_row(position=pos, ancestral_state=ancestral_state)

    tables.sort()
    return tables.tree_sequence()


# Example usage
ts = msprime.simulate(
    4, length=20, recombination_rate=0.2, mutation_rate=0.1, random_seed=2
)
ancestral_states = [
    "A",
    "C",
    "G",
    "T",
]  # Replace this list with your preferred ancestral states
new_ts = add_monomorphic_sites(ts, ancestral_states)


def floor_sites(ts):
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    positions = set()
    for tree in ts.trees():
        left, right = tree.interval
        for site in tree.sites():
            rounded = math.floor(site.position)
            if left <= rounded < right and rounded not in positions:
                positions.add(rounded)
                site_id = tables.sites.add_row(
                    rounded,
                    ancestral_state=site.ancestral_state,
                    metadata=site.metadata,
                )
                for mutation in site.mutations:
                    tables.mutations.add_row(
                        site=site_id,
                        node=mutation.node,
                        derived_state=mutation.derived_state,
                        parent=mutation.parent,
                        metadata=mutation.metadata,
                    )
    return tables.tree_sequence()


def get_full_genotypes(tree_sequence):
    num_samples = tree_sequence.num_samples // 2
    sequence_length = int(tree_sequence.sequence_length)
    full_genotypes = np.full(
        (num_samples, sequence_length), None, dtype=object
    )

    # Initialize with ancestral states
    for site in tree_sequence.sites():
        pos = int(site.position)
        ancestral_state = site.ancestral_state
        for i in range(num_samples):
            full_genotypes[i, pos] = (ancestral_state, ancestral_state)

    # Update for variant sites
    for variant in tree_sequence.variants():
        pos = int(variant.site.position)
        decoded_gt = np.array([variant.alleles[g] for g in variant.genotypes])
        decoded_gt = decoded_gt.reshape(-1, 2)
        for i in range(num_samples):
            full_genotypes[i, pos] = (decoded_gt[i][0], decoded_gt[i][1])

    return full_genotypes


g = demes.load("PG-SUI/pgsui/example_data/demography/example_demes2.yaml")
demography = msprime.Demography.from_demes(g)

# Minimal example for msprime simulation
sample_sets = [
    msprime.SampleSet(num_samples=3, population="pop1"),
    msprime.SampleSet(num_samples=2, population="pop2"),
]

anc_config = {
    "sequence_length": 20,
    "recombination_rate": 0.0,
    "demography": demography,
}

mut_config = {
    "rate": 1e-7,
    "model": msprime.GTR(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.25, 0.25, 0.25, 0.25],
    ),
}

ts = msprime.sim_ancestry(samples=sample_sets, **anc_config)
ts = msprime.sim_mutations(tree_sequence=ts, **mut_config)

# Floor the sites and add missing ones
ts = floor_sites(ts)
tables = ts.dump_tables()
new_sites = set(range(20)) - set(tables.sites.position)
tables.sort()
ts = tables.tree_sequence()

# Get the full genotypes
full_genotypes = get_full_genotypes(ts)
print(full_genotypes)
