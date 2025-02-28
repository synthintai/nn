/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "nn.h"

// Private functions

typedef float (*activation_function_t)(float a, bool derivative);

// Null activation function
static float activation_function_none(float a, bool derivative)
{
	return 0;
}

// Identity activation function
static float activation_function_identity(float a, bool derivative)
{
	if (derivative)
		return 1;
	return a;
}

// Linear activation function
static float activation_function_linear(float a, bool derivative)
{
	if (derivative)
		return 1;
	return a;
}

// Rectified Linear Unit (ReLU) activation function
static float activation_function_relu(float a, bool derivative)
{
	if (a >= 0)
		return (derivative ? 1 : a);
	return 0;
}

// Leaky Rectified Linear Unit (Leaky ReLU) activation function
static float activation_function_leaky_relu(float a, bool derivative)
{
	if (a > 0)
		return (derivative ? 1 : a);
	return (derivative ? 0.01 : a * 0.01f);
}

// Exponential Linear Unit (ELU) activation function
static float activation_function_elu(float a, bool derivative)
{
	if (a >= 0)
		return (derivative ? 1 : a);
	return (derivative ? activation_function_elu(a, false) : expf(a) - 1);
}

// Threshold activation function
static float activation_function_threshold(float a, bool derivative)
{
	if (derivative)
		return 0;
	return a > 0;
}

// Sigmoid activation function (aka Logistic, aka Soft Step)
static float activation_function_sigmoid(float a, bool derivative)
{
	if (derivative) {
		float f = activation_function_sigmoid(a, false);
		return(f * (1.0f - f));
	}
	return 1.0f / (1.0f + expf(-a));
}

// Sigmoid activation function using a lookup table
static float activation_function_sigmoid_fast(float a, bool derivative)
{
	// Sigmoid outputs
	const float s[] = {0.0,0.000045,0.000123,0.000335,0.000911,0.002473,0.006693,0.017986,0.047426,0.119203,0.268941,0.500000,0.731059,0.880797,0.952574,0.982014,0.993307,0.997527,0.999089,0.999665,0.999877,0.999955,1.0};
	// Derivative of the sigmoid
	const float ds[] = {0.0,0.000045,0.000123,0.000335,0.000910,0.002467,0.006648,0.017663,0.045177,0.104994,0.196612,0.250000,0.196612,0.104994,0.045177,0.017663,0.006648,0.002466,0.000910,0.000335,0.000123,0.000045,0.0};
	int index;
	float fraction = 0;

	index = floor(a) + 11;
	if (index < 0)
		index = 0;
	else if (index > 21)
		index = 21;
	else
		fraction = a - floor(a);
	if (derivative)
		return ds[index] + (ds[index + 1] - ds[index]) * fraction;
	return s[index] + (s[index + 1] - s[index]) * fraction;
}

// Tanh activation function
static float activation_function_tanh(float a, bool derivative)
{
	if (derivative)
		return 1.0 - activation_function_tanh(a, false) * activation_function_tanh(a, false);
	return (2.0 / (1.0 + expf(-2.0 * a))) - 1.0;
}

// Fast Tanh activation function
static float activation_function_tanh_fast(float a, bool derivative)
{
	if (derivative)
		return 1.0f / ((1.0f + abs(a)) * (1.0f + abs(a)));
	return a / (1.0f + abs(a));
}

// These must be in the same order as the enum activation_function_type
static activation_function_t activation_function[] = {
	activation_function_none,
	activation_function_identity,
	activation_function_linear,
	activation_function_relu,
	activation_function_leaky_relu,
	activation_function_elu,
	activation_function_threshold,
	activation_function_sigmoid,
	activation_function_sigmoid_fast,
	activation_function_tanh,
	activation_function_tanh_fast
};

// Computes the error given a cost function
static float error(float a, float b)
{
	return 0.5f * (a - b) * (a - b);
}

// Computes derivative of the error through the derivative of the cost function
static float error_derivative(float a, float b)
{
	return a - b;
}

static void forward_propagation(nn_t *nn)
{
	float sum;
	int i, j, k;

	// Calculate neuron values in each layer
	for (i = 1; i < nn->depth; i++) {
		for (j = 0; j < nn->width[i]; j++) {
			sum = 0;
			for (k = 0; k < nn->width[i - 1]; k++)
				sum += nn->neuron[i - 1][k] * nn->weight[i][j][k];
			sum += nn->bias[i];
			nn->neuron[i][j] = activation_function[nn->activation[i]](sum, false);
			// Store the preactivation value of this neuron for later use in backpropagation
			nn->preact[i][j] = sum;
		}
	}
}

// Public functions

nn_t *nn_init(void)
{
	nn_t *nn;

	nn = (nn_t *)malloc(sizeof(nn_t));
	if (NULL == nn)
		return NULL;
	nn->depth = 0;
	nn->width = NULL;
	nn->weight = NULL;
	nn->weight_adj = NULL;
	nn->neuron = NULL;
	nn->loss = NULL;
	nn->preact = NULL;
	nn->bias = NULL;
	nn->activation = NULL;
	return nn;
}

void nn_free(nn_t *nn)
{
	int layer, i;

	// There are no weights associated with the input layer, so we skip layer 0 and start at layer 1.
	for (layer = 1; layer < nn->depth; layer++) {
		for (i = 0; i < nn->width[layer]; i++) {
			free(nn->weight[layer][i]);
			free(nn->weight_adj[layer][i]);
		}
		free(nn->weight[layer]);
		free(nn->weight_adj[layer]);
	}
	// There are no neurons in the input layer, as the input array itself is used to store these values.
	for (layer = 1; layer < nn->depth; layer++) {
		free(nn->neuron[layer]);
		free(nn->loss[layer]);
		free(nn->preact[layer]);
	}
	free(nn->weight);
	free(nn->weight_adj);
	free(nn->neuron);
	free(nn->loss);
	free(nn->preact);
	free(nn->bias);
	free(nn->activation);
	free(nn->width);
	free(nn);
}

int nn_add_layer(nn_t *nn, int width, int activation, float bias)
{
	nn->depth++;
	nn->width = (uint32_t *)realloc(nn->width, nn->depth * sizeof(*nn->width));
	if (NULL == nn->width)
		return 1;
	nn->width[nn->depth - 1] = width;
	nn->activation = (uint8_t *)realloc(nn->activation, nn->depth * sizeof(*nn->activation));
	if (NULL == nn->activation)
		return 1;
	nn->activation[nn->depth - 1] = activation;
	nn->bias = (float *)realloc(nn->bias, nn->depth * sizeof(*nn->bias));
	if (NULL == nn->bias)
		return 1;
	nn->bias[nn->depth - 1] = bias;
	nn->neuron = (float **)realloc(nn->neuron, nn->depth * sizeof(float *));
	if (NULL == nn->neuron)
		return 1;
	nn->loss = (float **)realloc(nn->loss, nn->depth * sizeof(float *));
	if (NULL == nn->loss)
		return 1;
	nn->preact = (float **)realloc(nn->preact, nn->depth * sizeof(float *));
	if (NULL == nn->preact)
		return 1;
	if (nn->depth > 1) {
		nn->neuron[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
		if (NULL == nn->neuron[nn->depth - 1])
			return 1;
		nn->loss[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
		if (NULL == nn->loss[nn->depth - 1])
			return 1;
		nn->preact[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
		if (NULL == nn->preact[nn->depth - 1])
			return 1;
	}
	nn->weight = (float ***)realloc(nn->weight, (nn->depth) * sizeof(float **));
	if (NULL == nn->weight)
		return 1;
	nn->weight_adj = (float ***)realloc(nn->weight_adj, (nn->depth) * sizeof(float **));
	if (NULL == nn->weight_adj)
		return 1;
	if (nn->depth > 1) {
		nn->weight[nn->depth - 1] = (float **)malloc((nn->width[nn->depth - 1]) * sizeof(float *));
		if (NULL == nn->weight[nn->depth - 1])
			return 1;
		nn->weight_adj[nn->depth - 1] = (float **)malloc((nn->width[nn->depth - 1]) * sizeof(float *));
		if (NULL == nn->weight_adj[nn->depth - 1])
			return 1;
		for (int neuron = 0; neuron < nn->width[nn->depth - 1]; neuron++) {
			nn->weight[nn->depth - 1][neuron] = (float *)malloc((nn->width[nn->depth - 2]) * sizeof(float));
			if (NULL == nn->weight[nn->depth - 1][neuron])
				return 1;
			nn->weight_adj[nn->depth - 1][neuron] = (float *)malloc((nn->width[nn->depth - 2]) * sizeof(float));
			if (NULL == nn->weight_adj[nn->depth - 1][neuron])
				return 1;
			// Randomize the weights in this layer using uniform Xavier initialization
			// Range = +/- sqrt(6 / (#inputs + #outputs))
			for (int i = 0; i < nn->width[nn->depth - 2]; i++) {
				nn->weight[nn->depth - 1][neuron][i] = sqrtf(6.0f / (nn->width[nn->depth - 1] + nn->width[nn->depth - 2])) * 2.0f * (rand() / (float)RAND_MAX - 0.5f);
			}
		}
	}
	return 0;
}

// Trains a nn with a given input and target output at a specified learning rate.
// The rate (or step size) controls how far in the search space to move against the
// gradient in each iteration of the algorithm.
// Returns the total error between the target and the output of the neural network.
float nn_train(nn_t *nn, float *inputs, float *targets, float rate)
{
	float sum;
	int i, j, k;
	float err = 0;

	nn->neuron[0] = inputs;
	forward_propagation(nn);
	// Perform back propagation using gradient descent, which is an optimization algorithm that follows the
	// negative gradient of the objective function to find the minimum of the function.
	// Start at the output layer, and work backward toward the input layer, adjusting weights along the way.
	// Calculate the error aka loss aka delta at the output.
	// Select last layer (output layer)
	i = nn->depth - 1;
	for (j = 0; j < nn->width[i]; j++) {
		// Calculate the loss between the target and the outputs of the last layer
		nn->loss[i][j] = error_derivative(targets[j], nn->neuron[i][j]);
		err += error(targets[j], nn->neuron[i][j]);
	}
	// Calculate losses throughout the inner layers, not including layer 0 which can have no loss
	for (i = nn->depth - 2; i > 0 ; i--) {
		for (j = 0; j < nn->width[i]; j++) {
			sum = 0;
			for (k = 0; k < nn->width[i + 1]; k++)
				sum += nn->loss[i + 1][k] * activation_function[nn->activation[i]](nn->preact[i + 1][k], true) * nn->weight[i + 1][k][j];
			nn->loss[i][j] = sum;
		}
	}
	// Calculate the weight adjustments
	// The weights cannot be updated while back-propagating, because back propagating each layer depends on the next layer's weights.
	// So we save the weight adjustments in a temporary array and apply them all at once later.	
	for (i = nn->depth - 1; i > 0 ; i--)
		for (j = 0; j < nn->width[i]; j++)
			for (k = 0; k < nn->width[i - 1]; k++)
				nn->weight_adj[i][j][k] = nn->loss[i][j] * activation_function[nn->activation[i]](nn->preact[i][j], true) * nn->neuron[i - 1][k];
	// Apply the weight adjustments
	for (i = nn->depth - 1; i > 0 ; i--)
		for (j = 0; j < nn->width[i]; j++)
			for (k = 0; k < nn->width[i - 1]; k++)
				nn->weight[i][j][k] += nn->weight_adj[i][j][k] * rate;
	return err;
}

// Returns an output prediction given an input
float *nn_predict(nn_t *nn, float *inputs)
{
	nn->neuron[0] = inputs;
	forward_propagation(nn);
	// Return a pointer to the output layer
	return nn->neuron[nn->depth - 1];
}

float *nn_predict_quantized(nn_quantized_t *qmodel, float *input) {
	if (!qmodel || !input)
		return NULL;
	nn_t *original = qmodel->original_network;
	int depth = original->depth;
	float *activations = malloc(sizeof(float) * original->width[0]);
	memcpy(activations, input, sizeof(float) * original->width[0]);
	for (int layer = 1; layer < depth; layer++) {
		int curr_width = original->width[layer];
		float *new_activations = malloc(sizeof(float) * curr_width);
		for (int neuron = 0; neuron < curr_width; neuron++) {
			// Dequantize weights and compute dot product
			float sum = 0.0f;
			for (int w = 0; w < original->width[layer - 1]; w++) {
				float weight = qmodel->quantized_weights[layer][neuron][w] * qmodel->weight_scales[layer][neuron];
				sum += weight * activations[w];
			}
			// Dequantize bias
			float bias = qmodel->quantized_biases[layer][neuron] * qmodel->bias_scales[layer];
			sum += bias;
			// Apply activation
			new_activations[neuron] = activate(sum, original->activation[layer]);
		}
		free(activations);
		activations = new_activations;
	}
	return activations;
}

// Loads a neural net model file
nn_t *nn_load(char *path)
{
	FILE *file;
	nn_t *nn;
	int width = 0;
	int activation = ACTIVATION_FUNCTION_TYPE_NONE;
	float bias = 0;
	int layer, i, j;
	int depth;

	file = fopen(path, "r");
	if (NULL == file)
		return NULL;
	nn = nn_init();
	fscanf(file, "%d\n", &depth);
	for (i = 0; i < depth; i++) {
		fscanf(file, "%d %d %f\n", &width, &activation, &bias);
		if (nn_add_layer(nn, width, activation, bias) != 0) {
			fclose(file);
			return NULL;
		}
	}
	// Read in the weights
	for (layer = 1; layer < nn->depth; layer++)
		for (i = 0; i < nn->width[layer]; i++)
			for (j = 0; j < nn->width[layer - 1]; j++)
				fscanf(file, "%f\n", &nn->weight[layer][i][j]);
	fclose(file);
	return nn;
}

// Saves a neural net model file
int nn_save(nn_t *nn, char *path)
{
	int layer, i, j;
	FILE *file;

	file = fopen(path, "w");
	if (NULL == file)
		return 1;
	// File format:
	// depth
	// width, activation, bias
	// weight

	fprintf(file, "%ld\n", nn->depth);
	for (i = 0; i < nn->depth; i++)
		fprintf(file, "%ld %d %f\n", nn->width[i], nn->activation[i], nn->bias[i]);
	for (layer = 1; layer < nn->depth; layer++)
		for (i = 0; i < nn->width[layer]; i++)
			for (j = 0; j < nn->width[layer - 1]; j++)
				fprintf(file, "%f\n", nn->weight[layer][i][j]);
	fclose(file);
	return 0;
}

void nn_free_quantized(nn_quantized_t* quantized_network) {
	if (!quantized_network) return;

	for (int layer = 1; layer < quantized_network->original_network->depth; layer++) {
		int curr_width = quantized_network->original_network->width[layer];
		
		for (int neuron = 0; neuron < curr_width; neuron++) {
			free(quantized_network->quantized_weights[layer][neuron]);
		}
		free(quantized_network->quantized_weights[layer]);
		free(quantized_network->weight_scales[layer]);
		free(quantized_network->quantized_biases[layer]);
	}

	free(quantized_network->quantized_weights);
	free(quantized_network->weight_scales);
	free(quantized_network->quantized_biases);
	free(quantized_network->bias_scales);
	free(quantized_network);
}

nn_quantized_t* nn_load_quantized(const char* path) {
	FILE* file = fopen(path, "r");
	if (!file) {
		fprintf(stderr, "Failed to open quantized model file: %s\n", path);
		return NULL;
	}

	// Initialize quantized model structure
	nn_quantized_t* qmodel = malloc(sizeof(nn_quantized_t));
	if (!qmodel) {
		fclose(file);
		return NULL;
	}

	// Load original network structure
	qmodel->original_network = nn_init();
	int depth;
	
	// Read network depth
	if (fscanf(file, "%d\n", &depth) != 1) {
		fprintf(stderr, "Error reading network depth\n");
		goto error;
	}

	// Read layer configurations
	for (int i = 0; i < depth; i++) {
		int width, activation;
		float bias;
		if (fscanf(file, "%d %d %f\n", &width, &activation, &bias) != 3) {
			fprintf(stderr, "Error reading layer %d configuration\n", i+1);
			goto error;
		}
		if (nn_add_layer(qmodel->original_network, width, activation, bias) != 0) {
			fprintf(stderr, "Error adding layer %d\n", i+1);
			goto error;
		}
	}

	// Allocate quantization arrays
	int max_layers = qmodel->original_network->depth;
	qmodel->quantized_weights = malloc(sizeof(int8_t**) * max_layers);
	qmodel->weight_scales = malloc(sizeof(float*) * max_layers);
	qmodel->quantized_biases = malloc(sizeof(int8_t*) * max_layers);
	qmodel->bias_scales = malloc(sizeof(float) * max_layers);

	// Read weights and biases for each layer
	for (int layer = 1; layer < qmodel->original_network->depth; layer++) {
		int curr_width = qmodel->original_network->width[layer];
		int prev_width = qmodel->original_network->width[layer-1];

		// Allocate weight storage
		qmodel->quantized_weights[layer] = malloc(sizeof(int8_t*) * curr_width);
		qmodel->weight_scales[layer] = malloc(sizeof(float) * curr_width);

		for (int neuron = 0; neuron < curr_width; neuron++) {
			// Read weight scale
			if (fscanf(file, "%f\n", &qmodel->weight_scales[layer][neuron]) != 1) {
				fprintf(stderr, "Error reading scale for layer %d neuron %d\n", layer, neuron);
				goto error;
			}

			// Read quantized weights
			qmodel->quantized_weights[layer][neuron] = malloc(sizeof(int8_t) * prev_width);
			for (int w = 0; w < prev_width; w++) {
				if (fscanf(file, "%hhd\n", &qmodel->quantized_weights[layer][neuron][w]) != 1) {
					fprintf(stderr, "Error reading weight for layer %d neuron %d weight %d\n", layer, neuron, w);
					goto error;
				}
			}
		}

		// Read bias scale
		if (fscanf(file, "%f\n", &qmodel->bias_scales[layer]) != 1) {
			fprintf(stderr, "Error reading bias scale for layer %d\n", layer);
			goto error;
		}

		// Read quantized biases
		qmodel->quantized_biases[layer] = malloc(sizeof(int8_t) * curr_width);
		for (int neuron = 0; neuron < curr_width; neuron++) {
			if (fscanf(file, "%hhd\n", &qmodel->quantized_biases[layer][neuron]) != 1) {
				fprintf(stderr, "Error reading bias for layer %d neuron %d\n", layer, neuron);
				goto error;
			}
		}
	}

	fclose(file);
	return qmodel;

error:
	fclose(file);
	nn_free_quantized(qmodel);
	return NULL;
}

float activate(float value, int activation_type) {
	// Bounds check to prevent invalid access
	if (activation_type < 0 || activation_type >= (int)(sizeof(activation_function)/sizeof(activation_function[0]))) {
		return activation_function[ACTIVATION_FUNCTION_TYPE_NONE](value, false);
	}
	return activation_function[activation_type](value, false);
}

uint32_t nn_version(void)
{
	return (NN_VERSION_MAJOR << 24) | (NN_VERSION_MINOR << 16) | (NN_VERSION_PATCH << 8) | NN_VERSION_BUILD;
}
