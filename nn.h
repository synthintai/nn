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
#define NN_VERSION_BUILD	0

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
float nn_error(nn_t *nn, float *inputs, float *targets);
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);
float *nn_predict_quantized(nn_quantized_t* qmodel, float* input);
uint32_t nn_version(void);
int nn_remove_neuron(nn_t *nn, int layer, int neuron_index);
float nn_get_total_neuron_weight(nn_t *nn, int layer, int neuron_index);
bool nn_prune_lightest_neuron(nn_t *nn);
void nn_pool2d(char *src, char *dest, int filter_size, int stride, pooling_type_t pooling_type, int x_in_size, int y_in_size, int *x_out_size, int *y_out_size);
void nn_conv2d(char *src, char *dest, int8_t *kernel, int kernel_size, int stride, activation_function_type_t activation_function_type, int x_in_size, int y_in_size, int *x_out_size, int *y_out_size);

#endif /* NN_H */
