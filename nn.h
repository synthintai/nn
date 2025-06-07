/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_H
#define NN_H

// NN API Version
#define NN_VERSION_MAJOR	0
#define NN_VERSION_MINOR	1
#define NN_VERSION_PATCH	3
#define NN_VERSION_BUILD	2

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
	POOLING_TYPE_NONE = 0,
	POOLING_TYPE_MIN,
	POOLING_TYPE_MAX,
	POOLING_TYPE_AVG,
} pooling_type_t;

typedef struct {
	bool quantized;			// Indicates if the network is quantized
	uint8_t version_major;	// Major version of the network model
	uint8_t version_minor;	// Minor version of the network model
	uint8_t version_patch;	// Patch level of the network model
	uint8_t version_build;	// Build number of the network model
	uint32_t depth;			// Number of layers, including the input and the output layers
	uint32_t *width;		// Number of neurons in each layer (can vary from layer to layer)
	uint8_t *activation;	// Activation function used for each layer (can be different for each layer)
	float **neuron;			// Output value for each neuron in each layer
	float **loss;			// Error derivative for each neuron in each layer
	float **preact;			// Neuron values before activation function is applied for each neuron in each layer
	float **weight_scale;
	union {
		int8_t ***weight_quantized;	// Quantized weight for each neuron in each layer
		float ***weight;	// Weight for each neuron in each layer
	};
	float ***weight_adj;	// Adjustment of each weight for each neuron in each layer
	float *bias_scale;
	union {
		int8_t **bias_quantized; // Quantized bias for each neuron
		float **bias;		// Bias for each neuron
	};
} nn_t;

uint32_t nn_version(void);
nn_t *nn_init(void);
void nn_free(nn_t *nn);
int nn_add_layer(nn_t *nn, int width, int activation);
int nn_save_model(nn_t *nn, char *path);
nn_t *nn_load_model(char *path);
float nn_error(nn_t *nn, float *inputs, float *targets);
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);
int nn_remove_neuron(nn_t *nn, int layer, int neuron_index);
float nn_get_total_neuron_weight(nn_t *nn, int layer, int neuron_index);
bool nn_prune_lightest_neuron(nn_t *nn);
void nn_pool2d(char *src, char *dest, int filter_size, int stride, pooling_type_t pooling_type, int x_in_size, int y_in_size, int *x_out_size, int *y_out_size);
void nn_conv2d(char *src, char *dest, int8_t *kernel, int kernel_size, int stride, activation_function_type_t activation_function_type, int x_in_size, int y_in_size, int *x_out_size, int *y_out_size);

#endif /* NN_H */
