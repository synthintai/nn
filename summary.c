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

	nn = nn_load("model.txt");
	if (NULL == nn) {
		printf("Error: Missing or invalid model file.\n");
		return 1;
	}
        printf("Layer\tType\tWidth\tActvation\tBias\n");
        for (int i = 0; i < nn->depth; i++) {
                printf("%d\t%s\t%d\t%d\t%f\n", i, "dense", nn->width[i], nn->activation[i], nn->bias[i]);
        }
	return 0;
}
