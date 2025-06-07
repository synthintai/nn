/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "nn.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to the neural-net model to prune (e.g., "
           "model.txt)\n");
    return 1;
  }
  const char *model_path = argv[1];
  nn_t *nn = nn_load_model_ascii((char *)model_path);
  if (nn == NULL) {
    fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
    return 1;
  }
  nn_prune_lightest_neuron(nn);
  nn_save_model_ascii(nn, (char *)model_path);
  nn_free(nn);
  return 0;
}
