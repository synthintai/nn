/*
 * Neural Network library
 * Copyright (c) 2019-2024 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "nn.h"

int main(void)
{
	nn_t *model;

	model = nn_load("model.txt");
	if (NULL == model) {
		printf("Error: Missing or invalid model file.\n");
		return 1;
	}
	while(1);
	return 0;
}
