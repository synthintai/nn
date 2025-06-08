/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

 /*
 * Converts a quantized neural net model into a floating-point neural network.
 */

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "nn.h"

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <input_model> <output_model>\n", argv[0]);
    printf("  input_model:  path to the quantized neural network model\n");
    printf("  output_model: path where to save the dequantized model\n");
    return 1;
  }
  const char *input_model = argv[1];
  const char *output_model = argv[2];
  // Load the original floating-point network
  nn_t *network = nn_load_model_ascii((char *)input_model);
  if (!network) {
    fprintf(stderr, "Failed to load input model: %s\n", input_model);
    return 1;
  }
  // Dequantize the network
  if (!network->quantized) {
    fprintf(stderr, "Network has not been quantized\n");
    nn_free(network);
    return 1;
  }
  if (nn_dequantize(network) != 0) {
    fprintf(stderr, "Failed to dequantize network\n");
    nn_free(network);
    return 1;
  }
  // Save the dequantized network
  if (nn_save_model_ascii(network, output_model) != 0) {
    fprintf(stderr, "Failed to save dequantized model: %s\n", output_model);
    nn_free(network);
    return 1;
  }
  printf("Successfully dequantized model\n");
  printf("  Input:  %s\n", input_model);
  printf("  Output: %s\n", output_model);
  nn_free(network);
  return 0;
}
