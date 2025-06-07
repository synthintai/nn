Neural Network library for embedded systems

Copyright (c) 2019-2025 SynthInt Technologies, LLC

https://synthint.ai

SPDX-License-Identifier: Apache-2.0

## Overview

This is a lightweight neural network library for use in microcontrollers and embedded systems.

The code is divided into the following sections:

1. `nn.[ch]` - The neural net library, which can be pulled directly into your embedded project.

2. `data_prep.[ch]` - Data processing functions, used to read, parse, and shuffle training data.

3. `train.c` - An example of how to construct, train, and save a neural network model.

4. `test.c` - Evaluates model performance, comparing predictions to ground truth of seen vs. unseen data.

5. `predict.c` - Demonstrates how to use a trained neural network model in a target application to make predictions on new data.

6. `prune.c` - Removes least contributing neuron from a network to reduce model size and improve performance.

7. `quantize.c` - Converts a floating-point model to a 8-bit integer model. This reduces model size by about 66%.

8. `summary.c` - Describes a model.

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
The model can be further trained (or fine-tuned) simply by re-running the training program, which further trains a model file if it exists.

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

To prune the model (removes the least contributing neuron):

```
./prune
```

To quantize the trained model (which is floating point by default), run the following command:

```
./quantize model.txt model_quantized.txt
```

To test the quantized model:

```
./test_quantized
```

## Architecture

The network architecture is a fully connected feed-forward neural network. It is based on floating-point computation. The widths of each layer, the activation function to be used, and the bias for each layer is set as each successive layer is added to the network using the `nn_add_layer()` function call. Multiple layers may be added to construct a deep neural network.

## Demonstration

This embedded neural network library was used to power a handwritten character recognition application running on an STM32H7 microcontroller. Check it out here:

https://www.youtube.com/watch?v=cqjwSkrGtww

## Model File Format

The model file is saved as an ASCII file of floating-point values. The first line depicts the number of layers, inclusive of the input and output layers. The construct of each of those layers comprises the next set of lines, one line for each layer. The format of each line is width (in neurons), activation function, and bias. The remaining lines are the weights of each neuron in each layer, for all layers. Since there are no weights associated with the neurons in the input layer, these are skipped, and do not exist in the model file.

## Integration

To use this nn library in your own embedded system, it is only necessary to pull in the nn.c and nn.h files into your project. The other source files in the nn package are intended for data preparation for offline training, as well as examples of training and inference.

## License

Copyright (c) 2019-2025 SynthInt Technologies, LLC. All rights reserved.

Licensed under the [Apache License 2.0](./LICENSE).

## TODO

* Run cppcheck and fix all errors and warnings from static analysis
* Add nn_load_model_memory for embedded use
* Change pooling action to layer type that can be added to the model
* Change conv2d action to layer type that can be added to the model
* Add stride and padding parms to conv2d
* If using padding: feature_map_size = (N-F+2*P)/(S+1) <--the 2P is the padding
* Add dropout layer
* How to handle back prop for pooling layer?
* Number of kernels/filters dictates the number of output channels of a CNN layer
* Add auto-prune feature (to include cyclic training / pruning to achieve a desired minimum accuracy)
* Support CNN layer types in add_layer(), forward_propagate(), and nn_train().
* Add RNN feature
