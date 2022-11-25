/*
 * Neural Network library
 * Copyright (c) 2019-2022 Cole Design and Development, LLC
 * https://coledd.com
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
	int true_positive;

	// Recall a previously trained neural network model, inclusive of its weights
	model = nn_load("model.txt");
	if (NULL == model) {
		printf("Error: Missing or invalid model file.\n");
		return 1;
	}
	// Load training data into a data structure in memory
	data = data_load("train.csv", model->width[0], model->width[model->depth - 1]);
	num_samples = 0;
	true_positive = 0;
	for (int i = 0; i < data->num_rows; i++) {
		num_samples++;
		// Make an output prediction based upon new input data
		prediction = nn_predict(model, data->input[i]);
		for (int j = 0; j < model->width[model->depth - 1]; j++)
			if (data->target[i][j] > 0.5)
				if (prediction[j] > 0.5)
					true_positive++;
	}
	printf("Train: %d/%d = %2.2f%%\n", true_positive, num_samples, (true_positive * 100.0) / num_samples);
	data_free(data);
	// Load unseen data into a data structure in memory
	data = data_load("test.csv", model->width[0], model->width[model->depth - 1]);
	num_samples = 0;
	true_positive = 0;
	for (int i = 0; i < data->num_rows; i++) {
		num_samples++;
		// Make an output prediction based upon new input data
		prediction = nn_predict(model, data->input[i]);
		for (int j = 0; j < model->width[model->depth - 1]; j++)
			if (data->target[i][j] > 0.5)
				if (prediction[j] > 0.5)
					true_positive++;
	}
	printf("Test: %d/%d = %2.2f%%\n", true_positive, num_samples, (true_positive * 100.0) / num_samples);
	data_free(data);
	nn_free(model);
	return 0;
}

