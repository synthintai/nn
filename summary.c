/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include "nn.h"

// Activation‐function names must exactly match the enum in nn.h
static const char *activation_names[] = {
    "NONE",      "LINEAR",  "RELU",         "LEAKY_RELU", "ELU",
    "THRESHOLD", "SIGMOID", "SIGMOID_FAST", "TANH",       "TANH_FAST"};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a saved neural-net model (e.g., "
           "model.txt)\n");
    return 1;
  }
  const char *model_path = argv[1];
  nn_t *network = nn_load_model_ascii(model_path);
  if (!network) {
    fprintf(stderr, "Failed to load model: %s\n", model_path);
    return 1;
  }
  printf("Model Version:\t%u.%u.%u.%u\n", (unsigned)network->version_major,
         (unsigned)network->version_minor, (unsigned)network->version_patch,
         (unsigned)network->version_build);
  uint32_t lib_ver = nn_version();
  unsigned lib_major = (lib_ver >> 24) & 0xFF;
  unsigned lib_minor = (lib_ver >> 16) & 0xFF;
  unsigned lib_patch = (lib_ver >> 8) & 0xFF;
  unsigned lib_build = lib_ver & 0xFF;
  printf("NN Lib Version:\t%u.%u.%u.%u\n", lib_major, lib_minor, lib_patch,
         lib_build);
  if (network->quantized) {
    printf("Model Type:\tQuantized (fixed-point int8)\n");
  } else {
    printf("Model Type:\tFloating-point\n");
  }
  printf("Layer\tType\tWidth\tActivation\n");
  for (int i = 0; i < network->depth; i++) {
    const char *act_name = "UNKNOWN";
    uint8_t act_code = network->activation[i];
    if (act_code < (sizeof(activation_names) / sizeof(activation_names[0]))) {
      act_name = activation_names[act_code];
    }
    printf("%d\t%s\t%u\t%s\n", i, "dense", network->width[i], act_name);
  }
  nn_free(network);
  return 0;
}
