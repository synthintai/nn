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

correct = 0
data = b.data_load('train.csv', b.model.contents.width[0], b.model.contents.width[b.model.contents.depth-1])
for line in data:
	p = b.predict(line[:b.model.contents.width[0]])
	func = lambda x: round(x, 0)
	p_list = list(map(func, p[0:10]))
	target_list = line[256:266]
	if p_list == target_list:
		correct += 1
print(f'Train: {correct}/{len(data)} = {(correct / len(data)) * 100:0.2f}%')

correct = 0
data = b.data_load('test.csv', b.model.contents.width[0], b.model.contents.width[b.model.contents.depth-1])
for line in data:
	p = b.predict(line[:b.model.contents.width[0]])
	func = lambda x: round(x, 0)
	p_list = list(map(func, p[0:10]))
	target_list = line[256:266]
	if p_list == target_list:
		correct += 1
print(f'Test: {correct}/{len(data)} = {(correct / len(data)) * 100:0.2f}%')

