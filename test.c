/*
 * Neural Network library
 * Copyright (c) 2019 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nn.h"

// Data object
typedef struct {
	int num_rows;		// Number of rows of data
	int num_inputs;		// Number of inputs to neural network
	int num_outputs;	// Number of outputs from neural network
	float **input;		// 2D array of inputs
	float **target;		// 2D array of targets
} data_t;

// Allocates a data structure for the sample data
data_t *data_init(int num_rows, int num_inputs, int num_outputs)
{
	data_t *data=NULL;

	data=(data_t *)malloc(sizeof(data_t));
	if (NULL==data)
		return NULL;
	data->num_rows=num_rows;
	data->num_inputs=num_inputs;
	data->num_outputs=num_outputs;
	data->input=(float **)malloc(data->num_rows*sizeof(float *));
	if (NULL==data->input) {
		free(data);
		return NULL;
	}
	data->target=(float **)malloc(data->num_rows*sizeof(float *));
	if (NULL==data->target) {
		free(data->input);
		free(data);
		return NULL;
	}
	for (int i=0; i<data->num_rows; i++) {
		data->input[i]=(float *)malloc(data->num_inputs*sizeof(float));
		if (NULL==data->input[i]) {
			while(--i>=0) {
				free(data->input[i]);
				free(data->target[i]);
			}
			free(data->input);
			free(data->target);
			free(data);
			return NULL;
		}
		data->target[i]=(float *)malloc(data->num_outputs*sizeof(float));
		if (NULL==data->target[i]) {
			free(data->input[i]);
			while(--i>=0) {
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
void parse(data_t *data, char *line, int row)
{
	for (int column=0; column<(data->num_inputs+data->num_outputs); column++) {
		float val=atof(strtok(column==0?line:NULL, ","));
		if (column<data->num_inputs)
			data->input[row][column]=val;
		else
			data->target[row][column-data->num_inputs]=val;
	}
}

// Returns the number of lines in a file
int num_lines(FILE *file)
{
	int lines=0;
	int c=EOF;
	int previous_c='\n';

	while ((c=getc(file))!=EOF) {
		if (c=='\n')
			lines++;
		previous_c=c;
	}
	if (previous_c!='\n')
		lines++;
	rewind(file);
	return lines;
}

// Parses file from path getting all inputs and outputs for the neural network. Returns the data in a data structure.
data_t *load_data(char *path, int num_inputs, int num_outputs)
{
	int row;
	FILE *file;
	char *line=NULL;
	size_t len=0;
	int num_rows;
	data_t *data;

	file=fopen(path, "r");
	if (file==NULL) {
		printf("Error: Could not open %s\n", path);
		return NULL;
	}
	num_rows=num_lines(file);
	data=data_init(num_rows, num_inputs, num_outputs);
	if (NULL==data) {
		fclose(file);
		return NULL;
	}
	row=0;
	while (getline(&line, &len, file)!=-1)
		parse(data, line, row++);
	free(line);

	fclose(file);
	return data;
}

// Frees a data object
void data_free(data_t *data)
{
	for (int row=0; row<data->num_rows; row++) {
		free(data->input[row]);
		free(data->target[row]);
	}
	free(data->input);
	free(data->target);
	free(data);
}

// Randomly shuffles the rows of a data object
void shuffle(data_t *data)
{
	float *input, *output;

	for (int i=0; i<data->num_rows; i++) {
		int j=rand()%data->num_rows;
		// Swap target
		output=data->target[i];
		data->target[i]=data->target[j];
		data->target[j]=output;
		// Swap input
		input=data->input[i];
		data->input[i]=data->input[j];
		data->input[j]=input;
	}
}

int main(void)
{
	nn_t *model;
	data_t *data;
	float *prediction;
	int num_samples;
	int true_positive;

	// Recall a previously trained neural network model, inclusive of its weights
	model=nn_load("model.txt");
	if (NULL==model) {
		printf("Error: Missing or invalid model file.\n");
		return 1;
	}

	// Load training data into a data structure in memory
	data=load_data("train.csv", model->widths[0], model->widths[model->num_layers-1]);
	num_samples=0;
	true_positive=0;
	for (int i=0; i<data->num_rows; i++) {
		num_samples++;
		// Make an output prediction based upon new input data
		prediction=nn_predict(model, data->input[i]);
		for (int j=0; j<model->widths[model->num_layers-1]; j++)
			if (data->target[i][j]>0.5)
				if (prediction[j]>0.5)
					true_positive++;
	}
	printf("Train: %d/%d = %2.2f%%\n", true_positive, num_samples, (true_positive*100.0)/num_samples);
	data_free(data);

	// Load unseen data into a data structure in memory
	data=load_data("test.csv", model->widths[0], model->widths[model->num_layers-1]);
	num_samples=0;
	true_positive=0;
	for (int i=0; i<data->num_rows; i++) {
		num_samples++;
		// Make an output prediction based upon new input data
		prediction=nn_predict(model, data->input[i]);
		for (int j=0; j<model->widths[model->num_layers-1]; j++)
			if (data->target[i][j]>0.5)
				if (prediction[j]>0.5)
					true_positive++;
	}
	printf("Test: %d/%d = %2.2f%%\n", true_positive, num_samples, (true_positive*100.0)/num_samples);
	data_free(data);

	nn_free(model);
	return 0;
}

