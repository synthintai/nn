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
#define NN_VERSION_PATCH	2
#define NN_VERSION_BUILD	0

typedef enum activation_function_type {
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

typedef struct nn {
	uint32_t depth;		// Number of layers, including the input and the output layers
	uint32_t *width;	// Number of neurons in each layer (can vary from layer to layer)
	uint8_t *activation;// Activation function used for each layer (can be different for each layer)
	float *bias;		// Biases by layer (each layer can have its own bias)
	float **neuron;		// Output value for each neuron in each layer
	float **loss;		// Error derivative for each neuron in each layer
	float **preact;		// Neuron values before activation function is applied for each neuron in each layer
	float ***weight;	// Weight of each neuron in each layer
	float ***weight_adj;// Adjustment of each weight for each neuron in each layer
} nn_t;

typedef struct {
	nn_t* original_network;
	int8_t*** quantized_weights;
	float** weight_scales;
	int8_t** quantized_biases;
	float* bias_scales;
} nn_quantized_t;

nn_t *nn_init(void);
void nn_free(nn_t *nn);
void nn_free_quantized(nn_quantized_t* quantized_network);
int nn_add_layer(nn_t *nn, int width, int activation, float bias);
int nn_save_model(nn_t *nn, char *path);
nn_t *nn_load_model(char *path);
nn_quantized_t* nn_load_quantized(const char* path);
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);
float *nn_predict_quantized(nn_quantized_t* qmodel, float* input);
float activate(float value, int activation_type);
uint32_t nn_version(void);

#endif /* NN_H */
