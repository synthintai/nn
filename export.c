/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include "nn.h"

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <input_ascii_model> <output_binary_model>\n", argv[0]);
    printf("  input_model:  path to the ascii neural network model\n");
    printf("  output_model: path to the binary model to create\n");
    return 1;
  }
  const char *model_path = argv[1];
  nn_t *network = nn_load_model_ascii(model_path);
  if (!network) {
    fprintf(stderr, "Failed to load model: %s\n", model_path);
    return 1;
  }
  nn_save_model_binary(network, argv[2]);
  nn_free(network);
  return 0;
}
