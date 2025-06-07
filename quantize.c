/*
 * quantize.c
 *
 * Converts a floating‐point neural network (nn_t) into an 8‐bit quantized version
 * using the new unified nn_t definition from nn.h / nn.c.
 *
 * Usage: quantize <input_model> <output_model>
 *   input_model:  Path to the floating‐point neural network model
 *   output_model: Path where to save the quantized model
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include "nn.h"

//----------------------------------------------------------------
// Helper: Find minimum and maximum in a float array of length size.
//----------------------------------------------------------------
static void find_minmax(float *values, int size, float *min_val, float *max_val)
{
    *min_val = values[0];
    *max_val = values[0];
    for (int i = 1; i < size; i++) {
        if (values[i] < *min_val) *min_val = values[i];
        if (values[i] > *max_val) *max_val = values[i];
    }
}

//----------------------------------------------------------------
// Helper: Symmetric quantization of a single float value to int8.
//   value      : original float
//   scale      : floating point scale (max_abs / 127)
//   zero_point : typically 0 for symmetric quant.
// Returns: clamped int8 in [‐128, +127].
//----------------------------------------------------------------
static int8_t quantize_value(float value, float scale, float zero_point)
{
    float q = value / scale + zero_point;
    if (q > 127.0f)  return 127;
    if (q < -128.0f) return -128;
    return (int8_t)lroundf(q);
}

//----------------------------------------------------------------
// Create a new nn_t that holds quantized weights/biases
// copied from `network`. We set `quantized = true` and fill
// weight_quantized, weight_scale, bias_quantized, bias_scale.
// Note: We copy only depth, width[], activation[], and quant fields.
// Float‐based pointers (neuron, loss, preact, weight, weight_adj, bias)
// are left NULL in the returned nn_t.
//----------------------------------------------------------------
static nn_t *nn_quantize(nn_t *network)
{
    if (network == NULL) {
        return NULL;
    }

    // Allocate the new nn_t structure for the quantized network
    nn_t *qnet = (nn_t *)malloc(sizeof(nn_t));
    if (qnet == NULL) {
        return NULL;
    }

    // 1) Copy metadata
    qnet->quantized      = true;
    qnet->version_major  = network->version_major;
    qnet->version_minor  = network->version_minor;
    qnet->version_patch  = network->version_patch;
    qnet->version_build  = network->version_build;
    qnet->depth          = network->depth;

    // 2) Copy width[] array
    qnet->width = (uint32_t *)malloc(sizeof(uint32_t) * qnet->depth);
    if (qnet->width == NULL) {
        free(qnet);
        return NULL;
    }
    memcpy(qnet->width, network->width, sizeof(uint32_t) * qnet->depth);

    // 3) Copy activation[] array
    qnet->activation = (uint8_t *)malloc(sizeof(uint8_t) * qnet->depth);
    if (qnet->activation == NULL) {
        free(qnet->width);
        free(qnet);
        return NULL;
    }
    memcpy(qnet->activation, network->activation, sizeof(uint8_t) * qnet->depth);

    // 4) Null‐out all float‐based pointers
    qnet->neuron       = NULL;
    qnet->loss         = NULL;
    qnet->preact       = NULL;
    qnet->weight       = NULL;  // union member float ***weight
    qnet->weight_adj   = NULL;
    qnet->bias         = NULL;  // union member float **bias

    // 5) Allocate arrays for quantization fields:
    //    - weight_quantized:   (int8_t ***) per layer
    //    - weight_scale:       (float **)  per layer × per‐neuron
    //    - bias_quantized:     (int8_t **)  per layer × per‐neuron
    //    - bias_scale:         (float *)   one per layer
    qnet->weight_quantized = (int8_t ***)malloc(sizeof(int8_t **) * qnet->depth);
    qnet->weight_scale     = (float   **)malloc(sizeof(float *)   * qnet->depth);
    qnet->bias_quantized   = (int8_t **)malloc(sizeof(int8_t *)   * qnet->depth);
    qnet->bias_scale       = (float   *)malloc(sizeof(float)      * qnet->depth);
    if (!qnet->weight_quantized ||
        !qnet->weight_scale     ||
        !qnet->bias_quantized   ||
        !qnet->bias_scale) {
        // Allocation failure; clean up any partial allocations
        if (qnet->weight_quantized) free(qnet->weight_quantized);
        if (qnet->weight_scale)     free(qnet->weight_scale);
        if (qnet->bias_quantized)   free(qnet->bias_quantized);
        if (qnet->bias_scale)       free(qnet->bias_scale);
        free(qnet->activation);
        free(qnet->width);
        free(qnet);
        return NULL;
    }

    // Initialize layer‐0 fields (no weights/biases for input layer)
    qnet->weight_quantized[0] = NULL;
    qnet->weight_scale[0]     = NULL;
    qnet->bias_quantized[0]   = NULL;
    qnet->bias_scale[0]       = 0.0f;

    // 6) For each layer ≥1, allocate and quantize:
    for (int layer = 1; layer < (int)qnet->depth; layer++) {
        int prev_width = network->width[layer - 1];
        int curr_width = network->width[layer];

        // a) Allocate per‐layer arrays
        qnet->weight_quantized[layer] = (int8_t **)malloc(sizeof(int8_t *) * curr_width);
        qnet->weight_scale[layer]     = (float   *)malloc(sizeof(float)   * curr_width);
        qnet->bias_quantized[layer]   = (int8_t  *)malloc(sizeof(int8_t)  * curr_width);
        if (!qnet->weight_quantized[layer] ||
            !qnet->weight_scale[layer]     ||
            !qnet->bias_quantized[layer]) {
            // Clean up everything allocated so far
            for (int L = 1; L < layer; L++) {
                if (qnet->weight_quantized[L]) {
                    for (int n = 0; n < (int)qnet->width[L]; n++) {
                        free(qnet->weight_quantized[L][n]);
                    }
                    free(qnet->weight_quantized[L]);
                }
                free(qnet->weight_scale[L]);
                free(qnet->bias_quantized[L]);
            }
            free(qnet->weight_quantized);
            free(qnet->weight_scale);
            free(qnet->bias_quantized);
            free(qnet->bias_scale);
            free(qnet->activation);
            free(qnet->width);
            free(qnet);
            return NULL;
        }

        // b) Compute bias_scale for this layer (symmetric quant, using global per‐layer min/max)
        float min_bias, max_bias;
        find_minmax(network->bias[layer], curr_width, &min_bias, &max_bias);
        float layer_bias_scale = fmaxf(fabsf(min_bias), fabsf(max_bias)) / 127.0f;
        if (layer_bias_scale == 0.0f) {
            // If all biases are zero, avoid division by zero
            layer_bias_scale = 1e-8f;
        }
        qnet->bias_scale[layer] = layer_bias_scale;
        float bias_zero_point = 0.0f;  // symmetric quant always uses zero‐point = 0

        // c) For each neuron in this layer:
        for (int neuron = 0; neuron < curr_width; neuron++) {
            // c1) Allocate per‐neuron weight_quantized array
            qnet->weight_quantized[layer][neuron] = (int8_t *)malloc(sizeof(int8_t) * prev_width);
            if (!qnet->weight_quantized[layer][neuron]) {
                // Clean up up to this neuron
                for (int n = 0; n < neuron; n++) {
                    free(qnet->weight_quantized[layer][n]);
                }
                free(qnet->weight_quantized[layer]);
                free(qnet->weight_scale[layer]);
                free(qnet->bias_quantized[layer]);
                // Clean up earlier layers too
                for (int L = 1; L < layer; L++) {
                    for (int n = 0; n < (int)qnet->width[L]; n++) {
                        free(qnet->weight_quantized[L][n]);
                    }
                    free(qnet->weight_quantized[L]);
                    free(qnet->weight_scale[L]);
                    free(qnet->bias_quantized[L]);
                }
                free(qnet->weight_quantized);
                free(qnet->weight_scale);
                free(qnet->bias_quantized);
                free(qnet->bias_scale);
                free(qnet->activation);
                free(qnet->width);
                free(qnet);
                return NULL;
            }

            // c2) Find min/max of float weights for this neuron
            float min_w, max_w;
            find_minmax(network->weight[layer][neuron], prev_width, &min_w, &max_w);
            float neuron_weight_scale = fmaxf(fabsf(min_w), fabsf(max_w)) / 127.0f;
            if (neuron_weight_scale == 0.0f) {
                neuron_weight_scale = 1e-8f;
            }
            qnet->weight_scale[layer][neuron] = neuron_weight_scale;
            float weight_zero_point = 0.0f;  // symmetric quant

            // c3) Quantize each weight in this neuron’s row
            for (int w = 0; w < prev_width; w++) {
                float orig_w = network->weight[layer][neuron][w];
                qnet->weight_quantized[layer][neuron][w] = 
                    quantize_value(orig_w, neuron_weight_scale, weight_zero_point);
            }

            // c4) Quantize this neuron’s bias
            float orig_b = network->bias[layer][neuron];
            qnet->bias_quantized[layer][neuron] = 
                quantize_value(orig_b, layer_bias_scale, bias_zero_point);
        }
    }

    return qnet;
}

//----------------------------------------------------------------
// Save a quantized nn_t to disk, with a leading quantized‐flag.
// First line = 1 (since this is a quantized model).
// Then: depth, each layer’s width/activation, then weights & biases.
//----------------------------------------------------------------
static int save_quantized_model(const nn_t *qnet, const char *path)
{
    if (qnet == NULL || path == NULL) {
        return -1;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        return -1;
    }

    // 1) Write quantized flag (always 1 for a quantized model)
    fprintf(fp, "1\n");

    // 1a) Write model version (major, minor, patch, build)
    fprintf(fp, "%hhu %hhu %hhu %hhu\n", qnet->version_major, qnet->version_minor, qnet->version_patch, qnet->version_build);

    // 2) Write depth
    fprintf(fp, "%d\n", qnet->depth);

    // 3) Write width and activation for each layer
    for (int i = 0; i < (int)qnet->depth; i++) {
        fprintf(fp, "%u %u\n", (unsigned)qnet->width[i], (unsigned)qnet->activation[i]);
    }

    // 4) For each layer ≥1, write:
    //    [neuron 0..N-1] weight_scale[layer][neuron]
    //    [neuron 0..N-1, weight 0..M-1] quantized weight ints
    //    bias_scale[layer]
    //    [neuron 0..N-1] quantized bias ints
    for (int layer = 1; layer < (int)qnet->depth; layer++) {
        int prev_w = qnet->width[layer - 1];
        int curr_w = qnet->width[layer];

        // a) Per‐neuron weight_scale and weights
        for (int neuron = 0; neuron < curr_w; neuron++) {
            // Write weight_scale for this neuron
            fprintf(fp, "%f\n", qnet->weight_scale[layer][neuron]);
            // Write each quantized weight (int) for this neuron
            for (int w = 0; w < prev_w; w++) {
                fprintf(fp, "%d\n", (int)qnet->weight_quantized[layer][neuron][w]);
            }
        }
        // b) Write bias_scale for this entire layer
        fprintf(fp, "%f\n", qnet->bias_scale[layer]);

        // c) Write per‐neuron quantized bias
        for (int neuron = 0; neuron < curr_w; neuron++) {
            fprintf(fp, "%d\n", (int)qnet->bias_quantized[layer][neuron]);
        }
    }

    fclose(fp);
    return 0;
}

//----------------------------------------------------------------
// Free all memory owned by a quantized nn_t (qnet->quantized == true).
//----------------------------------------------------------------
static void free_quantized_network(nn_t *qnet)
{
    if (qnet == NULL) {
        return;
    }

    // 1) Free per‐layer quantized data
    for (int layer = 1; layer < (int)qnet->depth; layer++) {
        int curr_w = qnet->width[layer];
        // Free each neuron’s quantized weight array
        for (int neuron = 0; neuron < curr_w; neuron++) {
            free(qnet->weight_quantized[layer][neuron]);
        }
        free(qnet->weight_quantized[layer]);
        free(qnet->weight_scale[layer]);
        free(qnet->bias_quantized[layer]);
    }
    // 2) Free top‐level quant arrays
    free(qnet->weight_quantized);
    free(qnet->weight_scale);
    free(qnet->bias_quantized);
    free(qnet->bias_scale);

    // 3) Free width[] and activation[] (copied from original)
    free(qnet->activation);
    free(qnet->width);

    // 4) Finally, free the nn_t struct itself
    free(qnet);
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage: %s <input_model> <output_model>\n", argv[0]);
        printf("  input_model:  path to the floating point neural network model\n");
        printf("  output_model: path where to save the quantized model\n");
        return 1;
    }

    const char *input_model  = argv[1];
    const char *output_model = argv[2];

    // 1) Load the original floating‐point network
    nn_t *network = nn_load_model_ascii((char *)input_model);
    if (!network) {
        fprintf(stderr, "Failed to load input model: %s\n", input_model);
        return 1;
    }

    // 2) Quantize the network
    if (network->quantized) {
        fprintf(stderr, "Network has already been quantized\n");
        return 1;
    }
    nn_t *qnet = nn_quantize(network);
    if (!qnet) {
        fprintf(stderr, "Failed to quantize network\n");
        nn_free(network);
        return 1;
    }

    // 3) Save the quantized network (with leading “1\n” flag)
    if (save_quantized_model(qnet, output_model) != 0) {
        fprintf(stderr, "Failed to save quantized model: %s\n", output_model);
        free_quantized_network(qnet);
        nn_free(network);
        return 1;
    }

    printf("Successfully quantized model:\n");
    printf("  Input:  %s\n", input_model);
    printf("  Output: %s\n", output_model);

    // 4) Clean up
    free_quantized_network(qnet);
    nn_free(network);
    return 0;
}
