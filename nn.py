#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2022 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0

import ctypes
from ctypes import CDLL
from enum import Enum
import sys

ACTIVATION_FUNCTION_TYPE_NONE = 		0
ACTIVATION_FUNCTION_TYPE_IDENTITY =		1
ACTIVATION_FUNCTION_TYPE_LINEAR =		2
ACTIVATION_FUNCTION_TYPE_RELU =			3
ACTIVATION_FUNCTION_TYPE_LEAKY_RELU =	4
ACTIVATION_FUNCTION_TYPE_ELU =			5
ACTIVATION_FUNCTION_TYPE_THRESHOLD =	6
ACTIVATION_FUNCTION_TYPE_SIGMOID =		7
ACTIVATION_FUNCTION_TYPE_SIGMOID_FAST =	8
ACTIVATION_FUNCTION_TYPE_TANH =			9
ACTIVATION_FUNCTION_TYPE_TANH_FAST =	10

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

nn_init_c = lib.nn_init
nn_init_c.argtypes = []
nn_init_c.restype = ctypes.POINTER(struct_nn)

nn_free_c = lib.nn_free
nn_free_c.argtypes = [ctypes.POINTER(struct_nn)]
nn_free_c.restype = None

nn_add_layer_c = lib.nn_add_layer
nn_add_layer_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.c_int32, ctypes.c_int32, ctypes.c_float]
nn_add_layer_c.restype = ctypes.c_int32

nn_train_c = lib.nn_train
nn_train_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
nn_train_c.restype = ctypes.c_float

nn_predict_c = lib.nn_predict
nn_predict_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_float)]
nn_predict_c.restype = ctypes.POINTER(ctypes.c_float)

nn_load_c = lib.nn_load
nn_load_c.argtypes = [ctypes.POINTER(ctypes.c_char)]
nn_load_c.restype = ctypes.POINTER(struct_nn)

nn_save_c = lib.nn_save
nn_save_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_char)]
nn_save_c.restype = ctypes.c_int32

nn_version_c = lib.nn_version
nn_version_c.argtypes = []
nn_version_c.restype = ctypes.c_int32

def nn_init():
	return nn_init_c()

def nn_free(nn):
	nn_free_c(nn)

def nn_add_layer(nn, width, activation, bias):
	width_c = ctypes.c_int32(width)
	activation_c = ctypes.c_int32(activation)
	bias_c = ctypes.c_float(bias)
	return nn_add_layer_c(nn, width_c, activation_c, bias_c)

def nn_train(nn, inputs, targets, rate):
	input_width = nn.contents.width[0]
	output_width = nn.contents.width[nn.contents.depth-1]
	inputs_c_array = (ctypes.c_float * input_width)()
	for i, value in enumerate(inputs):
		inputs_c_array[i] = ctypes.c_float(value)
	targets_c_array = (ctypes.c_float * output_width)()
	for i, value in enumerate(targets):
		targets_c_array[i] = ctypes.c_float(value)
	rate_c = ctypes.c_float(rate)
	return nn_train_c(nn, inputs_c_array, targets_c_array, rate_c)

def nn_predict(nn, inputs):
	input_width = nn.contents.width[0]
	inputs_c_array = (ctypes.c_float * input_width)()
	for i, value in enumerate(inputs):
		inputs_c_array[i] = ctypes.c_float(value)
	return nn_predict_c(nn, inputs_c_array)

def nn_load(path):
	path_c = ctypes.c_char_p(path.encode('utf-8'))
	return nn_load_c(path_c)

def nn_save(nn, path):
	path_c = ctypes.c_char_p(path.encode('utf-8'))
	nn_save_c(nn, path_c)

def nn_version():
	return nn_version_c()

def data_load(path, num_inputs, num_outputs):
	try:
		f = open(path, 'r')
		data = f.readlines()
	except:
		print('Failed to read training data')
		return None
	stripped_lines = [s.rstrip(',\n') for s in data]
	fb = []
	for line in stripped_lines:
		line_split = line.split(',')
		fa = []
		try:
			fa = [float(i) for i in line_split]
		except:
			print('Error in training data format')
		fb.append(fa)
	return fb

nn = nn_init()
nn_add_layer(nn, 256, ACTIVATION_FUNCTION_TYPE_NONE, 0);
nn_add_layer(nn, 40, ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
nn_add_layer(nn, 40, ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
nn_add_layer(nn, 10, ACTIVATION_FUNCTION_TYPE_SIGMOID, 0);

data = data_load('train.csv', nn.contents.width[0], nn.contents.width[nn.contents.depth-1])
for j in range(1000):
	for line in data:
		nn_train(nn, line[:nn.contents.width[0]], line[nn.contents.width[0]:], 0.01)

nn_save(nn, 'model.txt')

nn_free(nn)

nn = nn_load('model.txt')
try:
	nn.contents
except ValueError:
	print('Cannot load model file')
	sys.exit(1)

true_positives = 0
data = data_load('train.csv', nn.contents.width[0], nn.contents.width[nn.contents.depth-1])
for line in data:
	p = nn_predict(nn, line[:nn.contents.width[0]])
	func = lambda x: round(x, 0)
	p_list = list(map(func, p[0:10]))
	target_list = line[256:266]
	if p_list == target_list:
		true_positives += 1
print(f'Train: {true_positives}/{len(data)} = {(true_positives / len(data)) * 100:0.2f}%')

true_positives = 0
data = data_load('test.csv', nn.contents.width[0], nn.contents.width[nn.contents.depth-1])
for line in data:
	p = nn_predict(nn, line[:nn.contents.width[0]])
	func = lambda x: round(x, 0)
	p_list = list(map(func, p[0:10]))
	target_list = line[256:266]
	if p_list == target_list:
		true_positives += 1
print(f'Test: {true_positives}/{len(data)} = {(true_positives / len(data)) * 100:0.2f}%')

