#!/bin/bash

# Define the file to search in
file="var_test.pbs.o9889153"
out_file="var.txt"

# Define the patterns to search for (lines starting with these prefixes)
pattern1="^Epoch: 500"
pattern2="^Total training time"
pattern3="^Total parameters"
pattern4="^variance a:"

# Extract and print lines matching the patterns
grep -E "$pattern1|$pattern2|$pattern3|$pattern4" "$file" > "$out_file"

