#!/usr/bin/env python

import random

# Read the original file
with open('samples.csv', 'r') as file:
    lines = file.readlines()

# Shuffle lines randomly
random.shuffle(lines)

# Calculate indices for splits
num_lines = len(lines)
train_end = int(num_lines * 0.8)
valid_end = train_end + int(num_lines * 0.1)

# Split lines into training, validation, and test sets
train_lines = lines[:train_end]
valid_lines = lines[train_end:valid_end]
test_lines = lines[valid_end:]

# Write to respective files
with open('train.csv', 'w') as file:
    file.writelines(train_lines)

with open('validation.csv', 'w') as file:
    file.writelines(valid_lines)

with open('test.csv', 'w') as file:
    file.writelines(test_lines)

print("Files created successfully:")
print(f"Train set: {len(train_lines)} lines")
print(f"Validation set: {len(valid_lines)} lines")
print(f"Test set: {len(test_lines)} lines")
