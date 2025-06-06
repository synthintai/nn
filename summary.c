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
    if (argc != 2) {
        printf("Usage: summary <model>\n");
        return 1;
    }
    char *model_path = argv[1];
    nn_t *network = nn_load_model(model_path);
    if (!network) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }
    printf("Model Version:\t%u.%u.%u.%u\n",
        (unsigned)network->version_major,
        (unsigned)network->version_minor,
        (unsigned)network->version_patch,
        (unsigned)network->version_build
    );
    uint32_t lib_ver = nn_version();
    unsigned lib_major = (lib_ver >> 24) & 0xFF;
    unsigned lib_minor = (lib_ver >> 16) & 0xFF;
    unsigned lib_patch = (lib_ver >>  8) & 0xFF;
    unsigned lib_build =  lib_ver        & 0xFF;
    printf("NN Lib Version:\t%u.%u.%u.%u\n", lib_major, lib_minor, lib_patch, lib_build);
    if (network->quantized) {
        printf("Model Type:\tQuantized (fixed-point int8)\n");
    } else {
        printf("Model Type:\tFloating-point\n");
    }
    printf("Layer\tType\tWidth\tActivation\n");
    for (int i = 0; i < network->depth; i++) {
        printf("%d\t%s\t%u\t%u\n", i, "dense", network->width[i], network->activation[i]);
    }
    nn_free(network);
    return 0;
}
