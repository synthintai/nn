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

print(f'Layer   Type    Width   Actvation       Bias')
for i in range(b.model.contents.depth):
	print(f'{i}\tdense\t{b.model.contents.width[i]}\t{b.ActivationFunctionType(b.model.contents.activation[i]).name}\t{b.model.contents.bias[i]}')
