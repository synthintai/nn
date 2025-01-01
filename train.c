/*
 * Neural Network library
 * Copyright (c) 2019-2024 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "nn.h"
#include "data_prep.h"

#define TARGET_TRAIN_ERROR	0.01

int main(void)
{
	// Tunable hyperparameters
	int num_inputs = 256;
	int num_outputs = 10;
	float learning_rate = 0.02f;
	float annealing = 1.0f;
	int epochs = 0;
	// End of tunable parameters
	data_t *data;
	nn_t *nn;
	int j;
	float train_error = 1.0f;

	// Set the random seed
	srand(time(0));
	// Load sample data into a data structure in memory
	data = data_load("train.csv", num_inputs, num_outputs);
	// Initialize a neural network model
	nn = nn_load("model.txt");
	if (NULL == nn) {
		printf("Creating new model.\n");
		nn = nn_init();
		// Construct the neural network, layer by layer
		nn_add_layer(nn, num_inputs, ACTIVATION_FUNCTION_TYPE_NONE, 0);
		nn_add_layer(nn, 100, ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
		nn_add_layer(nn, 50, ACTIVATION_FUNCTION_TYPE_LEAKY_RELU, 0);
		nn_add_layer(nn, num_outputs, ACTIVATION_FUNCTION_TYPE_SIGMOID, 0);
	} else {
		printf("Using existing model file\n");
		if ((nn->width[0] != num_inputs) || (nn->width[nn->depth - 1] != num_outputs))
		{
			printf("Error: Model is a different size.\n");
			return(1);
		}
	}
	printf("train error, learning_rate\n");
	while (train_error > TARGET_TRAIN_ERROR) {
		float total_error = 0.0f;
		// It is critical to shuffle training data before each epoch to properly train the model
		data_shuffle(data);
		for (j = 0; j < data->num_rows; j++) {
			float *input = data->input[j];
			float *target = data->target[j];
			total_error += nn_train(nn, input, target, learning_rate);
		}
		epochs++;
		train_error = total_error / data->num_rows;
		printf("%.5f, %.5f\n", train_error, learning_rate);
		learning_rate *= annealing;
		// Incremental save
		// Incremental saving of the neural network architecture and weights to a file so that it can be used later
		nn_save(nn, "model.txt");
	}
	data_free(data);
	nn_free(nn);
	printf("Final train error: %f\n", train_error);
	printf("Training epochs: %d\n", epochs);
	return 0;
}

