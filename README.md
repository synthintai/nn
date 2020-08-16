Neural Network library

Copyright (c) 2019-2020 Cole Design and Development, LLC

https://coledd.com

SPDX-License-Identifier: Apache-2.0

## Overview

This is a lightweight neural network library for use in microcontrollers and embedded systems.

The code is divided into the following sections:

1. `nn.[ch]` - The neural net library, which can be pulled directly into a project.

2. `data_prep.[ch]` - Data processing functions, used to read, parse, and shuffle sample data on which to train a model.

3. `train.c` - An example of how to construct, train, and save a neural network model.

4. `test.c` - Evaluates the model performance, comparing predictions to ground truth of seen vs. unseen data.

5. `predict.c` - Demonstrates how to use a trained neural network model in a target application to make predictions on new data.

## Features

With this library, neural networks of any width and depth may be constructed and trained. The following activation functions are supported:

* Identity
* Linear
* ReLU
* Leaky ReLU
* ELU
* Threshold
* Sigmoid
* TanH

Different activation functions may be assigned to each layer in the network.

A bias can be added to each layer independently.

## Instructions

To build the nn library and sample training and prediction programs, just type:
```
make
```


To train:
```
./train
```
The included example data is the Semeion Handwritten Digit Data Set from the UCI Machine Learning Repository at:

http://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit


To evaluate the model performance:
```
./test
```


To use the trained model:
```
./predict
```

## Architecture

The network architecture is a fully connected feed-forward neural network. It is based on floating-point computation. The widths of each layer, the activation function to be used, and the bias for each layer is set as each successive layer is added to the network using the `nn_add_layer()` function call. Multiple layers may be added to construct a deep neural network.

## Demonstration

This embedded neural network library was used to power a handwritten character recognition application running on an STM32H7 microcontroller. Check it out here:

https://www.youtube.com/watch?v=cqjwSkrGtww

## Model File Format

The model file is saved as an ASCII file of floating-point values. The first line depicts the number of layers, inclusive of the input and output layers. The construct of each of those layers comprises the next set of lines, one line for each layer. The format of each line is width (in neurons), activation function, and bias. The remaining lines are the weights of each neuron in each layer, for all layers. Since there are no weights associated with the neurons in the input layer, these are skipped, and do not exist in the model file.

## License

Copyright (c) 2019-2020 Cole Design and Development, LLC. All rights reserved.

Licensed under the [Apache License 2.0](./LICENSE).

