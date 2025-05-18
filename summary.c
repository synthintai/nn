/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "nn.h"

void print_usage()
{
	printf("Usage: summary <model>\n");
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		print_usage();
		return 1;
	}
	char* model = argv[1];
	nn_t* network = nn_load_model(model);
	if (!network) {
		fprintf(stderr, "Failed to load model: %s\n", model);
		return 1;
	}
	printf("Layer\tType\tWidth\tActvation\tBias\n");
	for (int i = 0; i < network->depth; i++) {
		printf("%d\t%s\t%d\t%d\t%f\n", i, "dense", network->width[i], network->activation[i], network->bias[i]);
	}
	return 0;
}

