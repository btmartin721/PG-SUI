#!/bin/zsh

INDIR="$HOME/PG-SUI/subset_phylip"
OUTDIR="iqtree_results_unique"
mkdir -p "$OUTDIR"

cd $OUTDIR

# Loop over all Phylip files
for file in "$INDIR"/*.phy; do
  # Create a unique prefix
  # file:r gets the basename without the prefix
  prefix="${file:t:r}"

  # Run IQ-TREE with fast mode
  iqtree3 -s "$file" -T 4 --seqtype DNA --fast --prefix "$prefix" -mset GTR -m MFP;
done

