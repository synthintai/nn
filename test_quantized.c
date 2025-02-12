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

int main(void) {
	nn_quantized_t *qmodel = NULL;
	data_t *data;
	float *prediction;
	int num_samples, correct, true_positive, false_positive;
	qmodel = nn_load_quantized("model_quantized.txt");
	if (!qmodel) {
		fprintf(stderr, "Error: Failed to load model file\n");
		return 1;
	}
	int input_size, output_size;
	input_size = qmodel->original_network->width[0];
	output_size = qmodel->original_network->width[qmodel->original_network->depth - 1];
	// Load training data into a data structure in memory
	data = data_load("train.csv", input_size, output_size);
	num_samples = 0;
	correct = 0;
	for (int i = 0; i < data->num_rows; i++) {
		num_samples++;
		prediction = nn_predict_quantized(qmodel, data->input[i]);
		true_positive = 0;
		false_positive = 0;
		for (int j = 0; j < output_size; j++) {
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
	data = data_load("test.csv", input_size, output_size);
	num_samples = 0;
	correct = 0;
	for (int i = 0; i < data->num_rows; i++) {
		num_samples++;
		prediction = nn_predict_quantized(qmodel, data->input[i]);
		true_positive = 0;
		false_positive = 0;
		for (int j = 0; j < output_size; j++) {
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
	nn_free_quantized(qmodel);
	return 0;
}

