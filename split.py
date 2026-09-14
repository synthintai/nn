#!/usr/bin/env python

# Shuffles a CSV file's lines and splits them into train.csv/validation.csv/
# test.csv in the given ratio. Doesn't look at what's in each line (column
# count, contents, anything) -- any line-oriented CSV works, not just the
# character-recognition example's samples.csv.

import argparse
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", help="Path to the CSV file to shuffle and split")
parser.add_argument("--train", type=float, default=0.8,
                     help="Fraction of lines written to train.csv (default: 0.8)")
parser.add_argument("--validation", type=float, default=0.1,
                     help="Fraction of lines written to validation.csv (default: 0.1); "
                          "the remainder goes to test.csv")
args = parser.parse_args()

if args.train < 0 or args.validation < 0 or args.train + args.validation > 1.0:
    parser.error("--train and --validation must each be >= 0 and sum to <= 1.0")

with open(args.input, "r") as file:
    lines = file.readlines()

random.shuffle(lines)

num_lines = len(lines)
train_end = int(num_lines * args.train)
valid_end = train_end + int(num_lines * args.validation)

train_lines = lines[:train_end]
valid_lines = lines[train_end:valid_end]
test_lines = lines[valid_end:]

with open("train.csv", "w") as file:
    file.writelines(train_lines)

with open("validation.csv", "w") as file:
    file.writelines(valid_lines)

with open("test.csv", "w") as file:
    file.writelines(test_lines)

print("Files created successfully:")
print(f"Train set: {len(train_lines)} lines")
print(f"Validation set: {len(valid_lines)} lines")
print(f"Test set: {len(test_lines)} lines")
