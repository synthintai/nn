/*
 * Neural Network library
 * Copyright (c) 2019-2024 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_H
#define NN_H

// NN API Version
#define NN_VERSION_MAJOR	0
#define NN_VERSION_MINOR	1
#define NN_VERSION_PATCH	0
#define NN_VERSION_BUILD	0

typedef enum activation_function_type {
	ACTIVATION_FUNCTION_TYPE_NONE = 0,
	ACTIVATION_FUNCTION_TYPE_IDENTITY,
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

typedef struct nn {
	int depth;			// Number of layers, including the input and the output layers
	int *width;			// Number of neurons in each layer (can vary from layer to layer)
	int *activation;	// Activation function used for each layer (can be different for each layer)
	float *bias;		// Biases by layer (each layer can have its own bias)
	float **neuron;		// Output value for each neuron in each layer
	float **loss;		// Error derivative for each neuron in each layer
	float **preact;		// Neuron values before activation function is applied for each neuron in each layer
	float ***weight;	// Weight of each neuron in each layer
	float ***weight_adj;// Adjustment of each weight for each neuron in each layer
} nn_t;

nn_t *nn_init(void);
void nn_free(nn_t *nn);
int nn_add_layer(nn_t *nn, int width, int activation, float bias);
int nn_save(nn_t *nn, char *path);
nn_t *nn_load(char *path);
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);
uint32_t nn_version(void);

#endif /* NN_H */
