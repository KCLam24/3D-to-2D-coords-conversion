#!/bin/bash

# Define the file to search in
file="density.pbs.o9916379"
out_file="dens_points.txt"


# Define the patterns to search for (lines starting with these prefixes)
pattern1="^Epoch: 500"
pattern2="^Total training time"
pattern3="^Total parameters"
pattern4="^variance a:"
pattern5="^Training data shape:"
# Extract and print lines matching the patterns
grep -E "$pattern1|$pattern2|$pattern3|$pattern4|$pattern5" "$file" > "$out_file"

