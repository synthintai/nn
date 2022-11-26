#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2022 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0

import sys
from nn import Nn

b = Nn()
b.load('model.txt')
try:
	b.model.contents
except ValueError:
	print('Cannot load model file')
	sys.exit(1)

true_positives = 0
data = b.data_load('train.csv', b.model.contents.width[0], b.model.contents.width[b.model.contents.depth-1])
for line in data:
	p = b.predict(line[:b.model.contents.width[0]])
	func = lambda x: round(x, 0)
	p_list = list(map(func, p[0:10]))
	target_list = line[256:266]
	if p_list == target_list:
		true_positives += 1
print(f'Train: {true_positives}/{len(data)} = {(true_positives / len(data)) * 100:0.2f}%')

true_positives = 0
data = b.data_load('test.csv', b.model.contents.width[0], b.model.contents.width[b.model.contents.depth-1])
for line in data:
	p = b.predict(line[:b.model.contents.width[0]])
	func = lambda x: round(x, 0)
	p_list = list(map(func, p[0:10]))
	target_list = line[256:266]
	if p_list == target_list:
		true_positives += 1
print(f'Test: {true_positives}/{len(data)} = {(true_positives / len(data)) * 100:0.2f}%')

