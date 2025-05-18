#!/bin/bash

# Check if the input file exists
if [ ! -f "samples.csv" ]; then
  echo "Error: samples.csv not found." >&2
  exit 1
fi

# Initialize an empty array to store lines
lines=()

# Read lines from file, preserving newlines
while IFS= read -r line; do
  lines+=("$line")
done < "samples.csv"


num_lines=${#lines[@]}

# Calculate split indices
train_end=$((num_lines * 8 / 10))
validation_end=$((num_lines * 9 / 10))

# Write to files
printf "%s\n" "${lines[@]:0:$train_end}" > train.csv
printf "%s\n" "${lines[@]:$train_end:$((validation_end - train_end))}" > validation.csv
printf "%s\n" "${lines[@]:$validation_end}" > test.csv
