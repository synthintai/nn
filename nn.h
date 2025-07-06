/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_H
#define NN_H

#include <stdbool.h>
#include <stdint.h>

// NN API Version
#define NN_VERSION_MAJOR 0
#define NN_VERSION_MINOR 1
#define NN_VERSION_PATCH 6
#define NN_VERSION_BUILD 0

typedef enum {
  NN_ERROR_NONE = 0,                    // No error
  NN_ERROR_INVALID_ARGUMENT = -1,       // Invalid argument passed to function
  NN_ERROR_OUT_OF_MEMORY = -2,          // Out of memory
  NN_ERROR_FILE_NOT_FOUND = -3,         // File not found
  NN_ERROR_FILE_READ = -4,              // Error reading file
  NN_ERROR_FILE_WRITE = -5,             // Error writing file
  NN_ERROR_UNSUPPORTED_LAYER = -6,      // Unsupported layer type
  NN_ERROR_UNSUPPORTED_ACTIVATION = -7, // Unsupported activation function
  NN_ERROR_UNSUPPORTED_POOLING = -8,    // Unsupported pooling type
  NN_ERROR_INVALID_LAYER = -9,          // Invalid layer index
  NN_ERROR_INVALID_ACTIVATION = -10,    // Invalid activation function index
  NN_ERROR_INVALID_POOLING = -11,       // Invalid pooling type index
  NN_ERROR_INVALID_INIT = -12,          // Invalid initialization type
  NN_ERROR_INVALID_CONFIG = -13,        // Invalid configuration for layer
  NN_ERROR_INVALID_VERSION = -14,       // Invalid version of the neural network model
  NN_ERROR_QUANTIZATION = -15,          // Error during quantization
  NN_ERROR_DEQUANTIZATION = -16,        // Error during dequantization
  NN_ERROR_TRAINING = -17,              // Error during training
  NN_ERROR_PREDICTION = -18,            // Error during prediction
  NN_ERROR_UNKNOWN = -19                // Unknown error
} nn_error_t;

typedef enum {
  ACTIVATION_FUNCTION_TYPE_NONE = 0,
  ACTIVATION_FUNCTION_TYPE_LINEAR,
  ACTIVATION_FUNCTION_TYPE_RELU,
  ACTIVATION_FUNCTION_TYPE_LEAKY_RELU,
  ACTIVATION_FUNCTION_TYPE_ELU,
  ACTIVATION_FUNCTION_TYPE_THRESHOLD,
  ACTIVATION_FUNCTION_TYPE_SIGMOID,
  ACTIVATION_FUNCTION_TYPE_SIGMOID_FAST,
  ACTIVATION_FUNCTION_TYPE_TANH,
  ACTIVATION_FUNCTION_TYPE_TANH_FAST
} activation_function_type_t;

typedef enum {
  LAYER_TYPE_NONE = 0,
  LAYER_TYPE_FC,         // Fully Connected Network Layer
  LAYER_TYPE_CNN,        // Convolutional Neural Network Layer - Not yet implemented
  LAYER_TYPE_POOL,       // Pooling Layer - Not yet implemented
  LAYER_TYPE_LSTM,       // Long Short-Term Memory Layer - Not yet implemented
  LAYER_TYPE_GRU,        // Gated Recurrent Unit Layer - Not yet implemented
  LAYER_TYPE_RNN,        // Recurrent Neural Network Layer - Not yet implemented
  LAYER_TYPE_ATTENTION,  // Attention Layer - Not yet implemented
  LAYER_TYPE_TRANSFORMER,// Transformer Layer - Not yet implemented
  LAYER_TYPE_INPUT,      // Input Layer
  LAYER_TYPE_OUTPUT      // Output Layer
} layer_type_t;

typedef enum {
  POOLING_TYPE_NONE = 0,
  POOLING_TYPE_MIN,
  POOLING_TYPE_MAX,
  POOLING_TYPE_AVG,
} pooling_type_t;

typedef enum {
  NN_INIT_NONE = 0,      // No initialization
  NN_INIT_ZEROS,         // Initialize weights to zero
  NN_INIT_ONES,          // Initialize weights to one
  NN_INIT_RANDOM,        // Random initialization
  NN_INIT_XAVIER,        // Xavier initialization
  NN_INIT_HE,            // He initialization
} nn_init_t;

typedef struct {
  uint16_t in_h;         // Input height
  uint16_t in_w;         // Input width
  uint16_t in_channels;  // # Input feature maps
  uint16_t out_channels; // # Output feature maps
  uint8_t kernel_size;   // Kernel width and height (square)
  uint8_t stride;        // Stride
  uint8_t padding;       // Padding (same for all sides)
  uint8_t dilation;      // Dilation (same for both dimensions)
  nn_init_t weight_init; // How to initialize weights
  nn_init_t bias_init;   // How to initialize biases
} cnn_t;

typedef struct {
  bool quantized;         // Indicates if the network is quantized
  uint8_t version_major;  // Major version of the network model
  uint8_t version_minor;  // Minor version of the network model
  uint8_t version_patch;  // Patch level of the network model
  uint8_t version_build;  // Build number of the network model
  uint32_t depth;         // Number of layers, including the input and the output layers
  uint32_t *width;        // Number of neurons in each layer (can vary from layer to layer)
  uint8_t *layer_type;    // Type of each layer
  uint8_t *activation;    // Activation function used for each layer
  float **neuron;         // Output value for each neuron in each layer
  float **loss;           // Error derivative for each neuron in each layer
  float **preact;         // Neuron values before activation function is applied for each neuron in each layer
  float **weight_scale;   // Scale for each weight in each layer
  float ***weight;        // Weight for each neuron in each layer
  int8_t ***weight_quantized; // Quantized weight for each neuron in each layer
  float ***weight_adj;    // Adjustment of each weight for each neuron in each layer
  float *bias_scale;      // Scale for each bias in each layer
  float **bias;           // Bias for each neuron
  int8_t **bias_quantized;// Quantized bias for each neuron
} nn_t;

uint32_t nn_version(void);
nn_t *nn_init(void);
void nn_free(nn_t *nn);
nn_error_t nn_add_layer(nn_t *nn, layer_type_t layer_type, int width, int activation, void *config);
nn_error_t nn_save_model_ascii(nn_t *nn, const char *path);
nn_error_t nn_save_model_binary(nn_t *nn, const char *path);
nn_t *nn_load_model_ascii(const char *path);
nn_t *nn_load_model_binary(const char *path);
float nn_error(nn_t *nn, float *inputs, float *targets);
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);
nn_error_t nn_remove_neuron(nn_t *nn, int layer, int neuron_index);
float nn_get_total_neuron_weight(nn_t *nn, int layer, int neuron_index);
bool nn_prune_lightest_neuron(nn_t *nn);
void nn_pool2d(char *src, char *dest, int filter_size, int stride, pooling_type_t pooling_type, int x_in, int y_in);
void nn_conv2d(nn_t *nn, int layer, int kernel_size, int stride, int x_in, int y_in);
nn_error_t nn_quantize(nn_t *nn);
nn_error_t nn_dequantize(nn_t *nn);

#endif /* NN_H */
