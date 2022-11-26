#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2022 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0

import ctypes
from ctypes import CDLL
from enum import Enum
import sys
from nn import Nn
import random

a = Nn()
a.version()
a.add_layer(256, Nn.ACTIVATION_FUNCTION_TYPE_NONE, 0);
a.add_layer(40, Nn.ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
a.add_layer(40, Nn.ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
a.add_layer(10, Nn.ACTIVATION_FUNCTION_TYPE_SIGMOID, 0);

data = a.data_load('train.csv', a.model.contents.width[0], a.model.contents.width[a.model.contents.depth-1])
for j in range(1000):
	random.shuffle(data)
	for line in data:
		a.train(line[:a.model.contents.width[0]], line[a.model.contents.width[0]:], 0.01)

a.save('model.txt')

