/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "data_prep.h"
#include "nn.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a saved neural-net model (e.g., model.txt)\n");
    return 1;
  }
  // Load a previously saved model (specified by the user)
  const char *model_path = argv[1];
  nn_t *model = nn_load_model_ascii((char *)model_path);
  if (model == NULL) {
    fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
    return 1;
  }
  // Load training data into memory
  data_t *data = data_load("train.csv", model->width[0], model->width[model->depth - 1]);
  if (data == NULL) {
    printf("Error: Could not load training data (train.csv).\n");
    nn_free(model);
    return 1;
  }
  // Evaluate accuracy on the training set
  int num_samples = 0;
  int correct = 0;
  for (int i = 0; i < data->num_rows; i++) {
    num_samples++;
    float *prediction = nn_predict(model, data->input[i]);
    int true_positive = 0;
    int false_positive = 0;
    // Binary decision: target >= 0.5 to be in the positive class, else negative
    for (int j = 0; j < model->width[model->depth - 1]; j++) {
      if (data->target[i][j] >= 0.5f) {
        if (prediction[j] >= 0.5f) {
          true_positive++;
        }
      } else {
        if (prediction[j] >= 0.5f) {
          false_positive++;
        }
      }
    }
    // A sample is correct if exactly one output neuron is “on” for a
    // true-positive and none for false-positive
    if ((true_positive == 1) && (false_positive == 0)) {
      correct++;
    }
    // Note: nn_predict manages its own buffers. Do NOT free(prediction) here.
  }
  printf("Train: %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0f) / (float)num_samples);
  data_free(data);
  // Load unseen (test) data and evaluate
  data = data_load("test.csv", model->width[0], model->width[model->depth - 1]);
  if (data == NULL) {
    printf("Error: Could not load test data (test.csv).\n");
    nn_free(model);
    return 1;
  }
  num_samples = 0;
  correct = 0;
  for (int i = 0; i < data->num_rows; i++) {
    num_samples++;
    float *prediction = nn_predict(model, data->input[i]);
    int true_positive = 0;
    int false_positive = 0;
    for (int j = 0; j < model->width[model->depth - 1]; j++) {
      if (data->target[i][j] >= 0.5f) {
        if (prediction[j] >= 0.5f) {
          true_positive++;
        }
      } else {
        if (prediction[j] >= 0.5f) {
          false_positive++;
        }
      }
    }
    if ((true_positive == 1) && (false_positive == 0)) {
      correct++;
    }
    // Again, do not free(prediction); nn_predict manages its own buffers.
  }
  printf("Test : %d/%d = %2.2f%%\n", correct, num_samples, (correct * 100.0f) / (float)num_samples);
  data_free(data);
  nn_free(model);
  return 0;
}
