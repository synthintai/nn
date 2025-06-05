/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include "nn.h"

void print_usage()
{
    printf("Usage: summary <model>\n");
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        print_usage();
        return 1;
    }
    char *model_path = argv[1];
    nn_t *network = nn_load_model(model_path);
    if (!network) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    // Print whether this is a floating‐point or quantized model
    if (network->quantized) {
        printf("Model Type:\tQuantized (fixed‐point int8)\n");
    } else {
        printf("Model Type:\tFloating‐point\n");
    }

    printf("Layer\tType\tWidth\tActivation\n");
    for (int i = 0; i < network->depth; i++) {
        printf("%d\t%s\t%d\t%d\n",
            i,
            "dense",
            (int)network->width[i],
            (int)network->activation[i]
        );
    }

    nn_free(network);
    return 0;
}
