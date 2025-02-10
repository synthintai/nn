#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2025 SynthInt Technologies, LLC
# https://synthint.ai
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

line = [0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0.0,
	0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0.0,
	0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0.0,
	0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0.0,
	0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0.0,
	0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0.0,
	0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0.0,
	0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0.0,
	0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0.0,
	0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0.0,
	0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0.0,
	0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0.0,
	0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0.0,
	0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0.0,
	0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0.0,
	0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0.0] 

p = b.predict(line[:b.model.contents.width[0]])
func = lambda x: round(x, 5)
p_list = list(map(func, p[0:10]))
for i in range(len(p_list)):
	print(f'{i}: {p_list[i]:0.5f}')

