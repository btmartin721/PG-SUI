#!/bin/zsh

INPUTDIR="validation_inputs"
OUTPUTDIR="validation_inputs_phylip_max150k"
MAXLOCI=150000

for file in "$INPUTDIR"/*.vcf.gz; do
    NLOCI=$(bgzip -dc "$file" | grep -v "^#" | wc -l)
    if (( NLOCI > MAXLOCI )); then
        echo "Skipping $file: contains $NLOCI loci, which exceeds the maximum of $MAXLOCI."
        continue
    fi
    python3 scripts/vcf2phy.py -v "$file" -o "$OUTPUTDIR"
done