#!/bin/bash

# Check if a command-line argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 num_sites"
    exit 1
fi

# Number of sites from the command-line argument
num_sites=$1

# Remove the output file if it exists
rm -f test14K.rate

# Write the header lines to the output file
awk '/^[0-9]/ {exit} 1' test.rate > test14K.rate

# Loop to write the data lines until we reach the required number of sites
seq_num=1
while (( seq_num <= num_sites )); do
    awk -v seq_num="$seq_num" -v num_sites="$num_sites" '
    BEGIN { OFS="\t"; }  # Output field separator
    /^[0-9]/ {  # If line starts with number
        if (seq_num > num_sites) { exit; }
        print seq_num, $2, $3, $4;  # Print sequence number and rest of the line
        seq_num++;
    }' test.rate >> test14K.rate
done

