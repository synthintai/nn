Neural Network library for embedded systems

Copyright (c) 2019-2025 SynthInt Technologies, LLC

https://synthint.ai

SPDX-License-Identifier: Apache-2.0

## Overview

This is a lightweight neural network library for use in microcontrollers and embedded systems.

The code is divided into the following sections:

1. `nn.[ch]` - The neural net library, which can be pulled directly into your embedded project.

2. `data_prep.[ch]` - Data processing functions, used to read, parse, and shuffle training data.

3. `dequantize.c` - Converts an 8-bit integer model into a floating point model.

4. `train.c` - An example of how to construct, train, and save a neural network model.

5. `test.c` - Evaluates model performance, comparing predictions to ground truth of seen vs. unseen data.

6. `predict.c` - Demonstrates how to use a trained neural network model in a target application to make predictions on new data.

7. `prune.c` - Removes least contributing neuron from a network to reduce model size and improve performance.

8. `quantize.c` - Converts a floating-point model to a 8-bit integer model.

9. `summary.c` - Describes a model file.

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
./train model.txt
```
The model can be further trained (or fine-tuned) simply by re-running the training program, which further trains a model file if it already exists.

The included example data is the MNIST data set.


To evaluate the model performance:
```
./test model.txt
```


To use the trained model:
```
./predict
```

To prune the model (this example removes the 10 least contributing neuron):

```
./prune model.txt 10
```

To quantize the trained model (which is floating point by default), run the following command:

```
./quantize model.txt model_quantized.txt
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

* Complete the CNN layer implementation
* Add nn_load_model_memory for embedded use
* Change pooling action to layer type that can be added to the model
* Add padding parms to conv2d
* If using padding: feature_map_size = (N-F+2*P)/(S+1) <--the 2P is the padding
* Add dropout layer type
* Handle back prop for pooling layer
* Add auto-prune feature (to include cyclic training / pruning to achieve a desired minimum accuracy)
* Add RNN feature
* Implement softmax layer
* Run cppcheck and fix all errors and warnings from static analysis
