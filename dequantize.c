/*
 * quantize.c
 *
 * Converts a quantized neural net model into a floating-point neural network
 * using the unified nn_t definition from nn.h / nn.c, in-place.
 *
 * Usage: dequantize <input_model> <output_model>
 *   input_model:  Path to the quantized neural network model
 *   output_model: Path where to save the dequantized model
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include "nn.h"

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: %s <input_model> <output_model>\n", argv[0]);
        printf("  input_model:  path to the quantized neural network model\n");
        printf("  output_model: path where to save the dequantized model\n");
        return 1;
    }
    const char *input_model  = argv[1];
    const char *output_model = argv[2];
    // 1) Load the original floating-point network
    nn_t *network = nn_load_model_ascii((char *)input_model);
    if (!network) {
        fprintf(stderr, "Failed to load input model: %s\n", input_model);
        return 1;
    }
    // 2) Dequantize the network
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
    // 3) Save the dequantized network (with leading "1\n" flag)
    if (nn_save_model_ascii(network, output_model) != 0) {
        fprintf(stderr, "Failed to save dequantized model: %s\n", output_model);
        nn_free(network);
        return 1;
    }
    printf("Successfully dequantized model\n");
    printf("  Input:  %s\n", input_model);
    printf("  Output: %s\n", output_model);
    // 4) Clean up
    nn_free(network);
    return 0;
}
