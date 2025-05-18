/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "nn.h"
#include "data_prep.h"

#define TARGET_VALIDATION_ERROR	0.07

int main(void)
{
	// Tunable hyperparameters
	int num_inputs = 256;
	int num_outputs = 10;
	float learning_rate = 0.02f;
	float annealing = 1.0f;
	int epochs = 0;
	// End of tunable parameters
	data_t *train_data;
	data_t *validation_data;
	nn_t *nn;
	int j;
	float train_error = 1.0f;
	float validation_error = 1.0f;
	float total_train_error = 0.0f;
	float total_validation_error = 0.0f;

	// Set the random seed
	srand(time(0));
	// Load the training data into a data structure in memory
	train_data = data_load("train.csv", num_inputs, num_outputs);
	if (NULL == train_data) {
		printf("Error: Could not load training data.\n");
		return(1);
	}
	// Load the validation data into a data structure in memory
	validation_data = data_load("validation.csv", num_inputs, num_outputs);
	if (NULL == validation_data) {
		printf("Error: Could not load validation data.\n");
		return(1);
	}
	// Initialize a neural network model
	nn = nn_load_model("model.txt");
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
			printf("Error: Model data is a different size than its meta data describes.\n");
			return(1);
		}
	}
	printf("train error, validation error, learning rate\n");
	while (validation_error > TARGET_VALIDATION_ERROR) {
		// It is critical to shuffle training data before each epoch to properly train the model
		data_shuffle(train_data);
		// Train on each row of training data
		total_train_error = 0.0f;
		for (j = 0; j < train_data->num_rows; j++) {
			float *input = train_data->input[j];
			float *target = train_data->target[j];
			total_train_error += nn_train(nn, input, target, learning_rate);
		}
		train_error = total_train_error / train_data->num_rows;
		// Check the model against the validation data
		total_validation_error = 0.0f;
		for (j = 0; j < validation_data->num_rows; j++) {
			float *input = validation_data->input[j];
			float *target = validation_data->target[j];
			total_validation_error += nn_error(nn, input, target);
		}
		validation_error = total_validation_error / validation_data->num_rows;
		epochs++;
		printf("%.5f, %.5f, %.5f\n", train_error, validation_error, learning_rate);
		learning_rate *= annealing;
		// At each training step, save the neural network architecture and weights to a file so that it can be used later
		nn_save_model(nn, "model.txt");
	}
	data_free(validation_data);
	data_free(train_data);
	nn_free(nn);
	printf("Final train error: %f\n", train_error);
	printf("Training epochs: %d\n", epochs);
	return 0;
}

