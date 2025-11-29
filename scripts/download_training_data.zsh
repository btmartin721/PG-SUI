#!/bin/zsh

OUTDIR="$HOME/PG-SUI/training_data"
MAXFILEMB=1024  # 1 GB
RPM=10  # Reduced to 10 for safety
# REPLACE WITH YOUR NEW TOKEN
APITOKEN="n4E2FFK9y4EP0xJkn8AfkBB4ctiCHuJ2vu25kkKVZT8"

mkdir -p "$OUTDIR"

echo "Downloading training data from Datadryad..."

# Removed the hanging flag. If you want to resume, use: --resume-from 500
python pgsui/training_data/datadryad_data_final_metadata.py \
    --download-files \
    --max-file-mb "$MAXFILEMB" \
    --rpm "$RPM" \
    --api-token "$APITOKEN" \
    --resume-from 189

echo "Download complete. Training data saved to $OUTDIR/"