#!/usr/bin/env python

# Neural Network library
# Copyright (c) 2019-2024 SynthInt Technologies, LLC
# https://synthint.ai
# SPDX-License-Identifier: Apache-2.0

import random
from nn import Nn

epochs = 10

a = Nn()
a.version()
a.add_layer(256, Nn.ActivationFunctionType.NONE.value, 0)
a.add_layer(100, Nn.ActivationFunctionType.LEAKY_RELU.value, 0)
a.add_layer(50, Nn.ActivationFunctionType.LEAKY_RELU.value, 0)
a.add_layer(10, Nn.ActivationFunctionType.SIGMOID.value, 0)

data = a.data_load('train.csv', a.model.contents.width[0], a.model.contents.width[a.model.contents.depth-1])
learning_rate = 0.02

print("train error, learning_rate")
for j in range(epochs):
    total_error = 0.0
    data = random.sample(data, len(data))
    for line in data:
        inputs = line[:a.model.contents.width[0]]
        targets = line[a.model.contents.width[0]:]
        error = a.train(inputs, targets, learning_rate)
        total_error += error
   
    avg_error = total_error / len(data)
    print(f"{avg_error:.5f},{learning_rate:.5f}")
    # Incremental saving of the neural network architecture and weights to a file so that it can be used later
    a.save('model.txt')
