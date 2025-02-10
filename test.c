/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "nn.h"
#include "data_prep.h"

int main(void)
{
	nn_t *model;
	data_t *data;
	float *prediction;
	int num_samples;
	int correct;
	int true_positive;
	int false_positive;

	// Recall a previously trained neural network model, inclusive of its weights
	model = nn_load("model.txt");
	if (NULL == model) {
		printf("Error: Missing or invalid model file.\n");
		return 1;
	}
	// Load training data into a data structure in memory
	data = data_load("train.csv", model->width[0], model->width[model->depth - 1]);
	num_samples = 0;
	correct = 0;
	for (int i = 0; i < data->num_rows; i++) {
		num_samples++;
		// Make an output prediction based upon new input data
		prediction = nn_predict(model, data->input[i]);
		true_positive = 0;
		false_positive = 0;
		for (int j = 0; j < model->width[model->depth - 1]; j++) {
			if (data->target[i][j] >= 0.5) {
				if (prediction[j] >= 0.5)
					true_positive++;
			} else {
				if (prediction[j] >= 0.5)
					false_positive++;
			}
		}
		if ((true_positive == 1) && (false_positive == 0))
			correct++;
	}
	printf("Train: %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0) / num_samples);
	data_free(data);
	// Load unseen data into a data structure in memory
	data = data_load("test.csv", model->width[0], model->width[model->depth - 1]);
	num_samples = 0;
	correct = 0;
	for (int i = 0; i < data->num_rows; i++) {
		num_samples++;
		// Make an output prediction based upon new input data
		prediction = nn_predict(model, data->input[i]);
		true_positive = 0;
		false_positive = 0;
		for (int j = 0; j < model->width[model->depth - 1]; j++) {
			if (data->target[i][j] >= 0.5) {
				if (prediction[j] >= 0.5)
					true_positive++;
			} else {
				if (prediction[j] >= 0.5)
					false_positive++;
			}
		}
		if ((true_positive == 1) && (false_positive == 0))
			correct++;
	}
	printf("Test: %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0) / num_samples);
	data_free(data);
	nn_free(model);
	return 0;
}

