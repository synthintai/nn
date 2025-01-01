#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2024 SynthInt Technologies, LLC
# https://synthint.ai
# SPDX-License-Identifier: Apache-2.0

import random
from nn import Nn

TARGET_TRAIN_ERROR = 0.01

# epochs = 10
train_error = 1.0

# Initialize a neural network model
a = Nn()
a.load("model.txt")
try:
	a.model.contents
	print("Using existing model file")
except:
	print("Creating new model.")
	a.__init__()
	a.add_layer(256, Nn.ActivationFunctionType.NONE.value, 0)
	a.add_layer(100, Nn.ActivationFunctionType.LEAKY_RELU.value, 0)
	a.add_layer(50, Nn.ActivationFunctionType.LEAKY_RELU.value, 0)
	a.add_layer(10, Nn.ActivationFunctionType.SIGMOID.value, 0)

data = a.data_load('train.csv', a.model.contents.width[0], a.model.contents.width[a.model.contents.depth-1])
learning_rate = 0.02

print("train error, learning_rate")
# for j in range(epochs):
while train_error > TARGET_TRAIN_ERROR:
	total_error = 0.0
	data = random.sample(data, len(data))
	for line in data:
		inputs = line[:a.model.contents.width[0]]
		targets = line[a.model.contents.width[0]:]
		total_error += a.train(inputs, targets, learning_rate)
	train_error = total_error / len(data)
	print(f"{train_error:.5f},{learning_rate:.5f}")
	# Incremental saving of the neural network architecture and weights to a file so that it can be used later
	a.save('model.txt')
