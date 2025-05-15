/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "nn.h"

int main(void)
{
	nn_t *nn;

	nn = nn_load_model("model.txt");
	if (NULL == nn) {
		printf("Error: Missing or invalid model file.\n");
		return 1;
	}
	nn_prune_lightest_neuron(nn);
	nn_save_model(nn, "model.txt");
	return 0;
}
