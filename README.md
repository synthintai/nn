Neural Network library for embedded systems

Copyright (c) 2019-2026 SynthInt Technologies, LLC

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
* GELU
* SiLU

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

The model can be saved in either of two formats:

* **ASCII** - a text file of floating-point values. The first line depicts the number of layers, inclusive of the input and output layers. The construct of each of those layers comprises the next set of lines, one line for each layer. The format of each line is width (in neurons), activation function, and bias. The remaining lines are the weights of each neuron in each layer, for all layers. Since there are no weights associated with the neurons in the input layer, these are skipped, and do not exist in the model file.

* **Binary** - a compact, raw binary encoding of the same information. Every binary model file begins with the 4-byte magic number `NNB1`, followed by the same fields the ASCII format stores (quantized flag, version, layer definitions, weights, and biases), written as raw integers/floats rather than text.

`nn_load_model()` reads a model file's first few bytes and dispatches to the ASCII or binary loader automatically based on the magic number, so any tool that calls it can open either kind of model file without knowing in advance which format it's in. `nn_save_model()` writes binary format when the destination path ends in `.bin` (case-insensitive) and ASCII format otherwise. `train`, `test`, `predict`, `prune`, `quantize`, `dequantize`, and `summary` all use these, so passing e.g. `model.bin` instead of `model.txt` is enough to train, evaluate, prune, (de)quantize, or run inference against a binary model file. `nn_load_model_ascii`/`nn_save_model_ascii` and `nn_load_model_binary`/`nn_save_model_binary` remain available for callers that need to force a specific format regardless of extension (as `export` and `import` do, to convert between the two).

## Integration

To use this nn library in your own embedded system, it is only necessary to pull in the nn.c and nn.h files into your project. The other source files in the nn package are intended for data preparation for offline training, as well as examples of training and inference.

## License

Copyright (c) 2019-2026 SynthInt Technologies, LLC. All rights reserved.

Licensed under the [Apache License 2.0](./LICENSE).

## TODO

* Add nn_load_model_memory for embedded use
* Add padding parms to conv2d
* If using padding: feature_map_size = (N-F+2*P)/(S+1) <--the 2P is the padding
* Add dropout layer type
* Add auto-prune feature (to include cyclic training / pruning to achieve a desired minimum accuracy)
* Add RNN feature
* Implement softmax layer
* Run cppcheck and fix all errors and warnings from static analysis
