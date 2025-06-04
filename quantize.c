#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "nn.h"

void print_usage()
{
	printf("Usage: quantize <input_model> <output_model>\n");
	printf("  input_model: path to the floating-point neural network model\n");
	printf("  output_model: path where to save the quantized model\n");
}

// Helper function to find min and max values in a layer
static void find_minmax(float *values, int size, float *min_val, float *max_val)
{
	*min_val = values[0];
	*max_val = values[0];
	for (int i = 1; i < size; i++) {
		if (values[i] < *min_val) *min_val = values[i];
		if (values[i] > *max_val) *max_val = values[i];
	}
}

// Helper function to quantize a single float value to int8
static int8_t quantize_value(float value, float scale, float zero_point)
{
	float quantized = value / scale + zero_point;
	if (quantized > 127) return 127;
		if (quantized < -128) return -128;
			return (int8_t)round(quantized);
}

nn_quantized_t *nn_quantize(nn_t* network)
{
	float min_weight, max_weight;
	float min_bias, max_bias;
	float weight_scale, bias_scale;
	float weight_zero_point, bias_zero_point;

	nn_quantized_t* quantized = malloc(sizeof(nn_quantized_t));
	if (!quantized)
		return NULL;
	quantized->original_network = network;
	// Allocate memory for quantized weights and scales
	quantized->weight = malloc(sizeof(int8_t**) * network->depth);
	quantized->weight_scale = malloc(sizeof(float*) * network->depth);
	quantized->bias = malloc(sizeof(int8_t**) * network->depth);
	quantized->bias_scale = malloc(sizeof(float) * network->depth);
	for (int layer = 1; layer < network->depth; layer++) {
		int prev_width = network->width[layer-1];
		int curr_width = network->width[layer];
		// Allocate memory for this layer
		quantized->weight[layer] = malloc(sizeof(int8_t*) * curr_width);
		quantized->weight_scale[layer] = malloc(sizeof(float) * curr_width);
		quantized->bias[layer] = malloc(sizeof(int8_t) * curr_width);
		// Quantize weights for each neuron in this layer
		// Calculate bias scale (handle zero bias case)
		find_minmax(network->bias[layer], curr_width, &min_bias, &max_bias);
		// Calculate scale and zero point for symmetric quantization
		bias_scale = (float)fmax(fabs(min_bias), fabs(max_bias)) / 127.0f;
		quantized->bias_scale[layer] = bias_scale;
		// For symmetric quantization
		bias_zero_point = 0.0f;
		for (int neuron = 0; neuron < curr_width; neuron++) {
			quantized->weight[layer][neuron] = malloc(sizeof(int8_t) * prev_width);
			// Find min/max for weights of this neuron
			find_minmax(network->weight[layer][neuron], prev_width, &min_weight, &max_weight);
			// Calculate scale and zero point for symmetric quantization
			weight_scale = (float)fmax(fabs(min_weight), fabs(max_weight)) / 127.0f;
			quantized->weight_scale[layer][neuron] = weight_scale;
			// For symmetric quantization
			weight_zero_point = 0.0f;
			// Quantize weights
			for (int w = 0; w < prev_width; w++) {
				quantized->weight[layer][neuron][w] = quantize_value(network->weight[layer][neuron][w], weight_scale, weight_zero_point);
			}
			// One bias per Neuron
			// Quantize biases
			quantized->bias[layer][neuron] = quantize_value(network->bias[layer][neuron], bias_scale, bias_zero_point);
		}
	}
	return quantized;
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		print_usage();
		return 1;
	}
	char* input_model = argv[1];
	char* output_model = argv[2];
	// Load the original network
	nn_t* network = nn_load_model(input_model);
	if (!network) {
		fprintf(stderr, "Failed to load input model: %s\n", input_model);
		return 1;
	}
	// Quantize the network (using symmetric 8-bit quantization)
	nn_quantized_t* quantized = nn_quantize(network);
	if (!quantized) {
		fprintf(stderr, "Failed to quantize network\n");
		nn_free(network);
		return 1;
	}
	// Save the quantized network
	if (nn_save_quantized(quantized, output_model) != 0) {
		fprintf(stderr, "Failed to save quantized model: %s\n", output_model);
		nn_free_quantized(quantized);
		nn_free(network);
		return 1;
	}
	printf("Successfully quantized model:\n");
	printf("  Input: %s\n", input_model);
	printf("  Output: %s\n", output_model);
	// Clean up
	nn_free_quantized(quantized);
	nn_free(network);
	return 0;
}
