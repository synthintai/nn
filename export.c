/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "nn.h"

int main(int argc, char *argv[])
{
  if (argc != 3 && argc != 4) {
    printf("Usage: %s <input_ascii_model> <output_model> [--inplace]\n", argv[0]);
    printf("  input_model:  path to the ascii neural network model\n");
    printf("  output_model: path to the binary model to create\n");
    printf("  --inplace:    write the zero-copy \"inplace\" format (magic NNP1,\n");
    printf("                see nn_load_model_inplace() in nn.h) instead of the\n");
    printf("                regular binary format (magic NNB1)\n");
    return 1;
  }
  bool inplace = false;
  if (argc == 4) {
    if (strcmp(argv[3], "--inplace") != 0) {
      fprintf(stderr, "Unknown option: %s\n", argv[3]);
      return 1;
    }
    inplace = true;
  }
  const char *model_path = argv[1];
  nn_t *network = nn_load_model_ascii(model_path);
  if (!network) {
    fprintf(stderr, "Failed to load model: %s\n", model_path);
    return 1;
  }
  nn_error_t err = inplace ? nn_save_model_inplace(network, argv[2]) : nn_save_model_binary(network, argv[2]);
  if (err != NN_ERROR_NONE) {
    fprintf(stderr, "Failed to save model: %s\n", argv[2]);
    nn_free(network);
    return 1;
  }
  nn_free(network);
  return 0;
}
