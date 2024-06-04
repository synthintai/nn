#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2024 SynthInt Technologies, LLC
# https://synthint.ai
# SPDX-License-Identifier: Apache-2.0

import ctypes
from ctypes import CDLL
from enum import Enum
import sys

class Nn:

	class ActivationFunctionType(Enum):
		NONE = 			0
		IDENTITY =		1
		LINEAR =		2
		RELU =			3
		LEAKY_RELU =	4
		ELU =			5
		THRESHOLD =		6
		SIGMOID =		7
		SIGMOID_FAST =	8
		TANH =			9
		TANH_FAST =		10

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

	init_c = lib.nn_init
	init_c.argtypes = []
	init_c.restype = ctypes.POINTER(struct_nn)

	free_c = lib.nn_free
	free_c.argtypes = [ctypes.POINTER(struct_nn)]
	free_c.restype = None

	add_layer_c = lib.nn_add_layer
	add_layer_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.c_int32, ctypes.c_int32, ctypes.c_float]
	add_layer_c.restype = ctypes.c_int32

	train_c = lib.nn_train
	train_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float]
	train_c.restype = ctypes.c_float

	predict_c = lib.nn_predict
	predict_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_float)]
	predict_c.restype = ctypes.POINTER(ctypes.c_float)

	load_c = lib.nn_load
	load_c.argtypes = [ctypes.POINTER(ctypes.c_char)]
	load_c.restype = ctypes.POINTER(struct_nn)

	save_c = lib.nn_save
	save_c.argtypes = [ctypes.POINTER(struct_nn), ctypes.POINTER(ctypes.c_char)]
	save_c.restype = ctypes.c_int32

	version_c = lib.nn_version
	version_c.argtypes = []
	version_c.restype = ctypes.c_int32

	def __init__(self):
		self.model = self.init_c()

	def __del__(self):
		self.free_c(self.model)

	def add_layer(self, width, activation, bias):
		width_c = ctypes.c_int32(width)
		activation_c = ctypes.c_int32(activation)
		bias_c = ctypes.c_float(bias)
		return self.add_layer_c(self.model, width_c, activation_c, bias_c)

	def train(self, inputs, targets, rate):
		input_width = self.model.contents.width[0]
		output_width = self.model.contents.width[self.model.contents.depth-1]
		inputs_c_array = (ctypes.c_float * input_width)()
		for i, value in enumerate(inputs):
			inputs_c_array[i] = ctypes.c_float(value)
		targets_c_array = (ctypes.c_float * output_width)()
		for i, value in enumerate(targets):
			targets_c_array[i] = ctypes.c_float(value)
		rate_c = ctypes.c_float(rate)
		return self.train_c(self.model, inputs_c_array, targets_c_array, rate_c)

	def predict(self, inputs):
		input_width = self.model.contents.width[0]
		inputs_c_array = (ctypes.c_float * input_width)()
		for i, value in enumerate(inputs):
			inputs_c_array[i] = ctypes.c_float(value)
		return self.predict_c(self.model, inputs_c_array)

	def load(self, path):
		path_c = ctypes.c_char_p(path.encode('utf-8'))
		self.free_c(self.model)
		self.model = self.load_c(path_c)

	def save(self, path):
		path_c = ctypes.c_char_p(path.encode('utf-8'))
		self.save_c(self.model, path_c)

	def version(self):
		return self.version_c()

	def data_load(self, path, num_inputs, num_outputs):
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

	def summary(self):
		print(f'Layer	Type	Width	Actvation	Bias')
		for i in range(self.model.contents.depth):
			print(f'{i}\tdense\t{self.model.contents.width[i]}\t{self.ActivationFunctionType(self.model.contents.activation[i]).name}\t{self.model.contents.bias[i]}')

