/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_prep.h"
#include "nn.h"

#define TARGET_VALIDATION_ERROR 0.07

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to the neural net model to load or create (e.g., model.txt)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Tunable hyperparameters
  int num_inputs = 256;
  int num_outputs = 10;
  float learning_rate = 0.02f;
  float annealing = 1.0f;
  // End of tunable parameters
  data_t *train_data;
  data_t *validation_data;
  nn_t *nn;
  int j;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  float total_train_error = 0.0f;
  float total_validation_err = 0.0f;
  // Set the random seed
  srand((unsigned)time(NULL));
  // Load the training data into memory
  train_data = data_load("train.csv", num_inputs, num_outputs);
  if (train_data == NULL) {
    printf("Error: Could not load training data (train.csv).\n");
    return 1;
  }
  // Load the validation data into memory
  validation_data = data_load("validation.csv", num_inputs, num_outputs);
  if (validation_data == NULL) {
    printf("Error: Could not load validation data (validation.csv).\n");
    data_free(train_data);
    return 1;
  }
  // Attempt to load an existing model
  nn = nn_load_model_ascii((char *)model_path);
  if (nn == NULL) {
    printf("Creating new model.\n");
    nn = nn_init();
    if (nn == NULL) {
      printf("Error: Failed to initialize new neural network.\n");
      data_free(train_data);
      data_free(validation_data);
      return 1;
    }
    // Construct the neural network, layer by layer
    nn_add_layer(nn, LAYER_TYPE_INPUT, num_inputs, ACTIVATION_FUNCTION_TYPE_NONE);
//    nn_add_layer(nn, LAYER_TYPE_CNN, 256, ACTIVATION_FUNCTION_TYPE_LINEAR);
    nn_add_layer(nn, LAYER_TYPE_FC, 100, ACTIVATION_FUNCTION_TYPE_RELU);
    nn_add_layer(nn, LAYER_TYPE_FC, 50, ACTIVATION_FUNCTION_TYPE_RELU);
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, num_outputs, ACTIVATION_FUNCTION_TYPE_SIGMOID);
  } else {
    printf("Using existing model file: %s\n", model_path);
    // Verify that model dimensions match expected inputs/outputs
    if ((nn->width[0] != (uint32_t)num_inputs) ||
        (nn->width[nn->depth - 1] != (uint32_t)num_outputs)) {
      printf("Error: Loaded model dimensions do not match expected %d→%d.\n", num_inputs, num_outputs);
      nn_free(nn);
      data_free(train_data);
      data_free(validation_data);
      return 1;
    }
  }
  printf("train error, validation error, learning rate\n");
  while (validation_error > TARGET_VALIDATION_ERROR) {
    // It is critical to shuffle training data before each epoch
    data_shuffle(train_data);
    // Train on each row of training data
    total_train_error = 0.0f;
    for (j = 0; j < train_data->num_rows; j++) {
      float *input = train_data->input[j];
      float *target = train_data->target[j];
      total_train_error += nn_train(nn, input, target, learning_rate);
    }
    train_error = total_train_error / (float)train_data->num_rows;
    // Check the model against the validation data
    total_validation_err = 0.0f;
    for (j = 0; j < validation_data->num_rows; j++) {
      float *input = validation_data->input[j];
      float *target = validation_data->target[j];
      total_validation_err += nn_error(nn, input, target);
    }
    validation_error = total_validation_err / (float)validation_data->num_rows;
    epochs++;
    printf("%.5f, %.5f, %.5f\n", train_error, validation_error, learning_rate);
    learning_rate *= annealing;
    // Save the neural network architecture and weights to the specified file
    nn_save_model_ascii(nn, (char *)model_path);
  }
  data_free(validation_data);
  data_free(train_data);
  nn_free(nn);
  printf("Final train error: %f\n", train_error);
  printf("Training epochs: %d\n", epochs);
  return 0;
}
