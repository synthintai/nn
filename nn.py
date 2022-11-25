#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2022 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0

import ctypes
from ctypes import CDLL
from enum import Enum
import sys

ACTIVATION_FUNCTION_TYPE_NONE = 0
ACTIVATION_FUNCTION_TYPE_IDENTITY = 1
ACTIVATION_FUNCTION_TYPE_LINEAR = 2
ACTIVATION_FUNCTION_TYPE_RELU = 3
ACTIVATION_FUNCTION_TYPE_LEAKY_RELU = 4
ACTIVATION_FUNCTION_TYPE_ELU = 5
ACTIVATION_FUNCTION_TYPE_THRESHOLD = 6
ACTIVATION_FUNCTION_TYPE_SIGMOID = 7
ACTIVATION_FUNCTION_TYPE_SIGMOID_FAST = 8
ACTIVATION_FUNCTION_TYPE_TANH = 9
ACTIVATION_FUNCTION_TYPE_TANH_FAST = 10

class struct_nn(ctypes.Structure):
	__slots__ = ['depth', 'width', 'activation', 'bias', 'neuron', 'loss', 'preact', 'weight', 'weight_adj']
	_fields_ = [('depth', ctypes.c_int32),
				('width', ctypes.POINTER(ctypes.c_int32)),
				('activation', ctypes.POINTER(ctypes.c_int32)),
				('bias', ctypes.POINTER(ctypes.c_float)),
				('neuron', ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
				('loss', ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
				('preact', ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
				('weight', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))),
				('weight_adj', ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))))
				]

 
lib = CDLL('./libnn.so')

nn_init = lib.nn_init
nn_init.argtypes = []
nn_init.restype = ctypes.POINTER(struct_nn)

nn_add_layer = lib.nn_add_layer
nn_add_layer.argtypes = [ctypes.POINTER(struct_nn), ctypes.c_int32, ctypes.c_int32, ctypes.c_float]
nn_add_layer.restype = ctypes.c_int32

nn_train = lib.nn_train
nn_train.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
nn_train.restype = ctypes.c_float

nn_predict = lib.nn_predict
nn_predict.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_float)]
nn_predict.restype = ctypes.POINTER(ctypes.c_float)

def init_nn():
	nn = nn_init()
	return nn

def add_layer(nn, width, activation, bias):
	width_c = ctypes.c_int32(width)
	activation_c = ctypes.c_int32(activation)
	bias_c = ctypes.c_float(bias)
	ret = nn_add_layer(nn, width_c, activation_c, bias_c)
	return ret

def train(nn, inputs, targets, rate):
	input_width = nn.contents.width[0]
	output_width = nn.contents.width[nn.contents.depth-1]
	inputs_c_array = (ctypes.c_float * input_width)()
	for i, value in enumerate(inputs):
		inputs_c_array[i] = ctypes.c_float(value)
	targets_c_array = (ctypes.c_float * output_width)()
	for i, value in enumerate(targets):
		targets_c_array[i] = ctypes.c_float(value)
	rate_c = ctypes.c_float(rate)
	err = nn_train(nn, inputs_c_array, targets_c_array, rate_c)
	return err

def predict(nn, inputs):
	input_width = nn.contents.width[0]
	inputs_c_array = (ctypes.c_float * input_width)()
	for i, value in enumerate(inputs):
		inputs_c_array[i] = ctypes.c_float(value)
	outputs = nn_predict(nn, inputs_c_array)
	return outputs

nn = init_nn()
add_layer(nn, 256, ACTIVATION_FUNCTION_TYPE_NONE, 0);
add_layer(nn, 40, ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
add_layer(nn, 40, ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
add_layer(nn, 10, ACTIVATION_FUNCTION_TYPE_SIGMOID, 0);

try:
	f = open('train.csv', 'r')
	data = f.readlines()
except:
	print("Failed to read training data")
	sys.exit(1)
stripped_lines = [s.rstrip(',\n') for s in data]
fb = []
for line in stripped_lines:
	line_split = line.split(',')
	fa = []
	try:
		fa = [float(i) for i in line_split]
	except:
		print('Error in training data format')
		pass
	fb.append(fa)

for j in range(10):
	for i in fb:
		train(nn, i[:256], i[256:], 0.1)

p = predict(nn, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

for i in range(nn.contents.width[nn.contents.depth-1]):
	print(round(p[i]))

