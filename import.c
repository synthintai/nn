/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "nn.h"

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: %s <input_binary_model> <output_ascii_model>\n", argv[0]);
        printf("  input_model:  path to the binary neural network model\n");
        printf("  output_model: path to the ascii model to create\n");
        return 1;
    }
    const char *model_path = argv[1];
    nn_t *network = nn_load_model_binary(model_path);
    if (!network) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }
    nn_save_model_ascii(network, argv[2]);
    nn_free(network);
    return 0;
}
