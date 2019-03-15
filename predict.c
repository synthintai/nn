/*
 * Neural Network library
 * Copyright (c) 2019 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include "nn.h"

int main(void)
{
	nn_t *model;

	// Test data upon which to make a prediction
	float in[]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0};
	// Recall a previously trained neural network model, inclusive of its weights
	model=nn_load("model.txt");
	if (NULL==model) {
		printf("Error: Missing or invalid model file.\n");
		return 1;
	}
	// Make an output prediction based upon new input data
	float *prediction=nn_predict(model, in);
	for (int i=0; i<model->widths[model->num_layers-1]; i++)
		printf("%d: %.5f\n", i, prediction[i]);
	nn_free(model);
	return 0;
}

