/*
 * Neural Network library
 * Copyright (c) 2019-2024 SynthInt Technologise, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_prep.h"

// Allocates a data structure for the sample data
data_t *data_init(int num_rows, int num_inputs, int num_outputs)
{
	data_t *data = NULL;

	data = (data_t *)malloc(sizeof(data_t));
	if (NULL == data)
		return NULL;
	data->num_rows = num_rows;
	data->num_inputs = num_inputs;
	data->num_outputs = num_outputs;
	data->input = (float **)malloc(data->num_rows * sizeof(float *));
	if (NULL == data->input) {
		free(data);
		return NULL;
	}
	data->target = (float **)malloc(data->num_rows * sizeof(float *));
	if (NULL == data->target) {
		free(data->input);
		free(data);
		return NULL;
	}
	for (int i = 0; i < data->num_rows; i++) {
		data->input[i] = (float *)malloc(data->num_inputs * sizeof(float));
		if (NULL == data->input[i]) {
			while(--i >= 0) {
				free(data->input[i]);
				free(data->target[i]);
			}
			free(data->input);
			free(data->target);
			free(data);
			return NULL;
		}
		data->target[i] = (float *)malloc(data->num_outputs * sizeof(float));
		if (NULL == data->target[i]) {
			free(data->input[i]);
			while(--i >= 0) {
				free(data->input[i]);
				free(data->target[i]);
			}
			free(data->input);
			free(data->target);
			free(data);
			return NULL;
		}
	}
	return data;
}

// Splits inputs and outputs into two separate data tables within the data structure
void data_parse(data_t *data, char *line, int row)
{
	for (int column = 0; column < (data->num_inputs + data->num_outputs); column++) {
		float val = atof(strtok(column == 0 ? line : NULL, ","));
		if (column < data->num_inputs)
			data->input[row][column] = val;
		else
			data->target[row][column - data->num_inputs] = val;
	}
}

// Returns the number of lines in a file
int data_num_lines(FILE *file)
{
	int lines = 0;
	int c = EOF;
	int previous_c = '\n';

	while ((c = getc(file)) != EOF) {
		if (c == '\n')
			lines++;
		previous_c = c;
	}
	if (previous_c != '\n')
		lines++;
	rewind(file);
	return lines;
}

// Parses file from path getting all inputs and outputs for the neural network. Returns the data in a data structure.
data_t *data_load(char *path, int num_inputs, int num_outputs)
{
	int row;
	FILE *file;
	char *line = NULL;
	size_t len = 0;
	int num_rows;
	data_t *data;

	file = fopen(path, "r");
	if (file == NULL) {
		printf("Error: Could not open %s\n", path);
		return NULL;
	}
	num_rows = data_num_lines(file);
	data = data_init(num_rows, num_inputs, num_outputs);
	if (NULL == data) {
		fclose(file);
		return NULL;
	}
	row = 0;
	while (getline(&line, &len, file) != -1)
		data_parse(data, line, row++);
	free(line);
	fclose(file);
	return data;
}

// Frees a data object
void data_free(data_t *data)
{
	for (int row = 0; row < data->num_rows; row++) {
		free(data->input[row]);
		free(data->target[row]);
	}
	free(data->input);
	free(data->target);
	free(data);
}

// Randomly shuffles the rows of a data object
void data_shuffle(data_t *data)
{
	float *input, *output;

	for (int i = 0; i < data->num_rows; i++) {
		int j = rand() % data->num_rows;
		// Swap target
		output = data->target[i];
		data->target[i] = data->target[j];
		data->target[j] = output;
		// Swap input
		input = data->input[i];
		data->input[i] = data->input[j];
		data->input[j] = input;
	}
}
