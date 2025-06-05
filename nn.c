/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>
#include <float.h>
#include "nn.h"

// Private functions

typedef float (*activation_function_t)(float a, bool derivative);

// Null activation function
static float activation_function_none(float a, bool derivative)
{
    return 0;
}

// Linear activation function (aka identity activation function)
static float activation_function_linear(float a, bool derivative)
{
    if (derivative)
        return 1;
    return a;
}

// Rectified Linear Unit (ReLU) activation function
static float activation_function_relu(float a, bool derivative)
{
    if (a >= 0)
        return (derivative ? 1 : a);
    return 0;
}

// Leaky Rectified Linear Unit (Leaky ReLU) activation function
static float activation_function_leaky_relu(float a, bool derivative)
{
    if (a > 0)
        return (derivative ? 1 : a);
    return (derivative ? 0.01f : a * 0.01f);
}

// Exponential Linear Unit (ELU) activation function
static float activation_function_elu(float a, bool derivative)
{
    if (a >= 0)
        return (derivative ? 1 : a);
    return (derivative ? activation_function_elu(a, false) : expf(a) - 1);
}

// Threshold activation function
static float activation_function_threshold(float a, bool derivative)
{
    if (derivative)
        return 0;
    return a > 0;
}

// Sigmoid activation function (aka Logistic, aka Soft Step)
static float activation_function_sigmoid(float a, bool derivative)
{
    if (derivative) {
        float f = activation_function_sigmoid(a, false);
        return (f * (1.0f - f));
    }
    return 1.0f / (1.0f + expf(-a));
}

// Sigmoid activation function using a lookup table
static float activation_function_sigmoid_fast(float a, bool derivative)
{
    // Sigmoid outputs
    const float s[] = {
        0.0f, 0.000045f, 0.000123f, 0.000335f, 0.000911f, 0.002473f, 0.006693f, 0.017986f,
        0.047426f, 0.119203f, 0.268941f, 0.500000f, 0.731059f, 0.880797f, 0.952574f, 0.982014f,
        0.993307f, 0.997527f, 0.999089f, 0.999665f, 0.999877f, 0.999955f, 1.0f
    };
    // Derivative of the sigmoid
    const float ds[] = {
        0.0f, 0.000045f, 0.000123f, 0.000335f, 0.000910f, 0.002467f, 0.006648f, 0.017663f,
        0.045177f, 0.104994f, 0.196612f, 0.250000f, 0.196612f, 0.104994f, 0.045177f, 0.017663f,
        0.006648f, 0.002466f, 0.000910f, 0.000335f, 0.000123f, 0.000045f, 0.0f
    };
    int index;
    float fraction = 0;

    if (a < -11.0f)
        a = -11.0f;
    else if (a > 11.0f)
        a = 11.0f;
    index = (int)floorf(a) + 11;
    if (index < 0) {
        index = 0;
    } else if (index > 21) {
        index = 21;
    } else {
        fraction = a - floorf(a);
    }
    if (derivative) {
        return ds[index] + (ds[index + 1] - ds[index]) * fraction;
    }
    return s[index] + (s[index + 1] - s[index]) * fraction;
}

// Tanh activation function
static float activation_function_tanh(float a, bool derivative)
{
    if (derivative)
        return 1.0f - activation_function_tanh(a, false) * activation_function_tanh(a, false);
    return (2.0f / (1.0f + expf(-2.0f * a))) - 1.0f;
}

// Fast Tanh activation function
static float activation_function_tanh_fast(float a, bool derivative)
{
    if (derivative)
        return 1.0f / ((1.0f + fabsf(a)) * (1.0f + fabsf(a)));
    return a / (1.0f + fabsf(a));
}

// These must be in the same order as the enum activation_function_type
static activation_function_t activation_function[] = {
    activation_function_none,
    activation_function_linear,
    activation_function_relu,
    activation_function_leaky_relu,
    activation_function_elu,
    activation_function_threshold,
    activation_function_sigmoid,
    activation_function_sigmoid_fast,
    activation_function_tanh,
    activation_function_tanh_fast
};

// Computes the error given a cost function
// The loss function is a basic mean-square error (MSE)
static float error(float a, float b)
{
    return 0.5f * (a - b) * (a - b);
}

// Computes derivative of the error through the derivative of the cost function
static float error_derivative(float a, float b)
{
    return a - b;
}

static void forward_propagation(nn_t *nn)
{
    float sum;
    int i, j, k;

    // Calculate neuron values in each layer
    for (i = 1; i < (int)nn->depth; i++) {
        for (j = 0; j < (int)nn->width[i]; j++) {
            sum = 0.0f;
            // Dot‐product: previous layer output * weight
            for (k = 0; k < (int)nn->width[i - 1]; k++) {
                if (nn->quantized)
			        sum += nn->neuron[i - 1][k] * nn->weight_quantized[i][j][k] * nn->weight_scale[i][j];
		        else
			        sum += nn->neuron[i - 1][k] * nn->weight[i][j][k];
            }
            // Add bias
            if (nn->quantized)
                sum += nn->bias_quantized[i][j] * nn->bias_scale[i];
	        else
                sum += nn->bias[i][j];

            // Apply activation
            nn->neuron[i][j] = activation_function[nn->activation[i]](sum, false);

            // Cache pre‐activation for backprop
            nn->preact[i][j] = sum;
        }
    }
}

// Public functions

uint32_t nn_version(void)
{
    return ( (uint32_t)NN_VERSION_MAJOR << 24 ) |
           ( (uint32_t)NN_VERSION_MINOR << 16 ) |
           ( (uint32_t)NN_VERSION_PATCH <<  8 ) |
             (uint32_t)NN_VERSION_BUILD;
}

nn_t *nn_init(void)
{
    nn_t *nn = (nn_t *)malloc(sizeof(nn_t));
    if (nn == NULL)
        return NULL;

    // Mark as non‐quantized by default
    nn->quantized       = false;
    // Populate version from macros
    nn->version_major   = NN_VERSION_MAJOR;
    nn->version_minor   = NN_VERSION_MINOR;
    nn->version_patch   = NN_VERSION_PATCH;
    nn->version_build   = NN_VERSION_BUILD;

    nn->depth           = 0;
    nn->width           = NULL;
    nn->activation      = NULL;

    // Floats‐only pointers are NULL initially
    nn->neuron          = NULL;
    nn->loss            = NULL;
    nn->preact          = NULL;
    nn->weight          = NULL;    // union member float ***weight
    nn->weight_adj      = NULL;
    nn->bias            = NULL;    // union member float **bias

    // Quantization‐related pointers start NULL
    nn->weight_quantized = NULL;
    nn->weight_scale    = NULL;
    nn->bias_quantized  = NULL;
    nn->bias_scale      = NULL;

    return nn;
}

void nn_free(nn_t *nn)
{
    if (nn == NULL)
        return;

    // Free weight‐ and bias‐related arrays layer by layer
    // There are no weights/biases for layer 0, so start from layer 1.
    for (int layer = 1; layer < (int)nn->depth; layer++) {
        // Free each neuron's weight and weight_adj in this layer
        for (int i = 0; i < (int)nn->width[layer]; i++) {
            free(nn->weight[layer][i]);
            free(nn->weight_adj[layer][i]);
        }
        free(nn->weight[layer]);
        free(nn->weight_adj[layer]);

        // Free the bias array for this layer
        free(nn->bias[layer]);
    }

    // Free neuron, loss, preact arrays (no entry for layer 0 beyond input pointer)
    for (int layer = 1; layer < (int)nn->depth; layer++) {
        free(nn->neuron[layer]);
        free(nn->loss[layer]);
        free(nn->preact[layer]);
    }

    // Free top‐level pointers
    free(nn->weight);
    free(nn->weight_adj);
    free(nn->neuron);
    free(nn->loss);
    free(nn->preact);
    free(nn->activation);
    free(nn->width);

    // If quantization fields were ever allocated, free them
    if (nn->weight_scale != NULL) {
        free(nn->weight_scale);
    }
    if (nn->bias_scale != NULL) {
        free(nn->bias_scale);
    }
    // Note: nn->weight_quantized and nn->bias_quantized would need to be freed
    // if someone sets nn->quantized = true and allocates them. But since our
    // public API no longer exposes any quant routines, we assume they remain NULL.

    free(nn);
}

int nn_add_layer(nn_t *nn, int width, int activation)
{
    // Increase depth by one
    nn->depth++;
    // Reallocate the width array
    nn->width = (uint32_t *)realloc(nn->width, nn->depth * sizeof(*nn->width));
    if (nn->width == NULL)
        return 1;
    nn->width[nn->depth - 1] = (uint32_t)width;

    // Reallocate the activation array
    nn->activation = (uint8_t *)realloc(nn->activation, nn->depth * sizeof(*nn->activation));
    if (nn->activation == NULL)
        return 1;
    nn->activation[nn->depth - 1] = (uint8_t)activation;

    // Reallocate neuron, loss, preact arrays
    nn->neuron = (float **)realloc(nn->neuron, nn->depth * sizeof(float *));
    if (nn->neuron == NULL)
        return 1;
    nn->loss = (float **)realloc(nn->loss, nn->depth * sizeof(float *));
    if (nn->loss == NULL)
        return 1;
    nn->preact = (float **)realloc(nn->preact, nn->depth * sizeof(float *));
    if (nn->preact == NULL)
        return 1;

    // For layer 0, we do not allocate neuron/loss/preact (input is provided externally)
    if (nn->depth > 1) {
        // Allocate neuron, loss, preact arrays for this new layer
        nn->neuron[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
        if (nn->neuron[nn->depth - 1] == NULL)
            return 1;
        nn->loss[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
        if (nn->loss[nn->depth - 1] == NULL)
            return 1;
        nn->preact[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
        if (nn->preact[nn->depth - 1] == NULL)
            return 1;
    }

    // Reallocate top‐level weight, weight_adj, and bias arrays
    nn->weight = (float ***)realloc(nn->weight, (nn->depth) * sizeof(float **));
    if (nn->weight == NULL)
        return 1;
    nn->weight_adj = (float ***)realloc(nn->weight_adj, (nn->depth) * sizeof(float **));
    if (nn->weight_adj == NULL)
        return 1;
    nn->weight_scale = (float **)realloc(nn->weight_scale, (nn->depth) * sizeof(float *));
    if (nn->weight_scale == NULL)
        return 1;
    nn->bias = (float **)realloc(nn->bias, (nn->depth) * sizeof(float *));
    if (nn->bias == NULL)
        return 1;
    nn->bias_scale = (float *)realloc(nn->bias_scale, (nn->depth) * sizeof(float));
    if (nn->bias_scale == NULL)
        return 1;

    if (nn->depth > 1) {
        // Allocate per‐neuron pointers in this new layer
        nn->weight[nn->depth - 1] = (float **)malloc(nn->width[nn->depth - 1] * sizeof(float *));
        if (nn->weight[nn->depth - 1] == NULL)
            return 1;
        nn->weight_adj[nn->depth - 1] = (float **)malloc(nn->width[nn->depth - 1] * sizeof(float *));
        if (nn->weight_adj[nn->depth - 1] == NULL)
            return 1;
        nn->weight_scale[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
        if (nn->weight_scale[nn->depth - 1] == NULL)
            return 1;
        nn->bias[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
        if (nn->bias[nn->depth - 1] == NULL)
            return 1;

        // Initialize weights, weight_adj, and biases for each neuron in this layer
        for (int neuron = 0; neuron < (int)nn->width[nn->depth - 1]; neuron++) {
            // Allocate the weight‐vector for this neuron
            nn->weight[nn->depth - 1][neuron] = (float *)malloc(nn->width[nn->depth - 2] * sizeof(float));
            if (nn->weight[nn->depth - 1][neuron] == NULL)
                return 1;
            // Allocate the weight_adj vector for this neuron
            nn->weight_adj[nn->depth - 1][neuron] = (float *)malloc(nn->width[nn->depth - 2] * sizeof(float));
            if (nn->weight_adj[nn->depth - 1][neuron] == NULL)
                return 1;

            // Xavier (Glorot) initialization for each weight
            float range = sqrtf(6.0f / (nn->width[nn->depth - 1] + nn->width[nn->depth - 2]));
            for (int i = 0; i < (int)nn->width[nn->depth - 2]; i++) {
                nn->weight[nn->depth - 1][neuron][i] = range * 2.0f * ((rand() / (float)RAND_MAX) - 0.5f);
            }

            // Initialize bias = 0
            nn->bias[nn->depth - 1][neuron] = 0.0f;
        }
    }

    return 0;
}

// Returns the total error of the network given a set of inputs and target outputs
float nn_error(nn_t *nn, float *inputs, float *targets)
{
    int i, j;
    float err = 0.0f;

    // Layer 0's neuron pointers simply reference the input array
    nn->neuron[0] = inputs;
    forward_propagation(nn);

    // Sum MSE on the final (output) layer
    i = (int)nn->depth - 1;
    for (j = 0; j < (int)nn->width[i]; j++) {
        err += error(targets[j], nn->neuron[i][j]);
    }
    return err;
}

// Trains a nn with a given input and target output at a specified learning rate.
// The rate (or step size) controls how far in the search space to move against the
// gradient in each iteration of the algorithm.
// Returns the total error between the target and the output of the neural network.
float nn_train(nn_t *nn, float *inputs, float *targets, float rate)
{
    float sum;
    int i, j, k;

    // 1) Forward pass
    nn->neuron[0] = inputs;
    forward_propagation(nn);

	// Perform back propagation using gradient descent, which is an optimization algorithm that follows the
	// negative gradient of the objective function to find the minimum of the function.
	// Start at the output layer, and work backward toward the input layer, adjusting weights along the way.
	// Calculate the error aka loss aka delta at the output.

    // 2) Compute output‐layer loss
    i = (int)nn->depth - 1;
    for (j = 0; j < (int)nn->width[i]; j++) {
        nn->loss[i][j] = error_derivative(targets[j], nn->neuron[i][j]);
    }

    // 3) Backpropagate loss into earlier layers
    for (i = nn->depth - 2; i > 0; i--) {
        for (j = 0; j < (int)nn->width[i]; j++) {
            sum = 0.0f;
            for (k = 0; k < (int)nn->width[i + 1]; k++) {
		// Apply the derivative of the activation function for the next layer’s neurons
                sum += nn->loss[i + 1][k] *
                       activation_function[nn->activation[i + 1]](nn->preact[i + 1][k], true) *
                       nn->weight[i + 1][k][j];
            }
		// The chain rule dictates that we should multiply the summed loss by the derivative of the activation at the current neuron, 
		// not only during weight updates, but immediately when calculating loss[i][j].
            nn->loss[i][j] = sum * activation_function[nn->activation[i]](nn->preact[i][j], true);
        }
    }

    // 4) Update biases (gradient descent step)
    for (i = 1; i < (int)nn->depth; i++) {
        for (j = 0; j < (int)nn->width[i]; j++) {
            nn->bias[i][j] += nn->loss[i][j] * rate;
        }
    }
	// Calculate the weight adjustments - However, their update is delayed until after full backprop traversal.
	// The weights cannot be updated while back-propagating, because back propagating each layer depends on the next layer's weights.
	// So we save the weight adjustments in a temporary array and apply them all at once later.	
    // 5) Compute weight adjustments (store in weight_adj)
    for (i = (int)nn->depth - 1; i > 0; i--) {
        for (j = 0; j < (int)nn->width[i]; j++) {
            for (k = 0; k < (int)nn->width[i - 1]; k++) {
                nn->weight_adj[i][j][k] = nn->loss[i][j] * nn->neuron[i - 1][k];
            }
        }
    }

    // 6) Apply weight adjustments
    for (i = (int)nn->depth - 1; i > 0; i--) {
        for (j = 0; j < (int)nn->width[i]; j++) {
            for (k = 0; k < (int)nn->width[i - 1]; k++) {
                nn->weight[i][j][k] += nn->weight_adj[i][j][k] * rate;
            }
        }
    }

    // Return the post‐update error
    return nn_error(nn, inputs, targets);
}

// Returns an output prediction given an input
float *nn_predict(nn_t *nn, float *inputs)
{
    nn->neuron[0] = inputs;
    forward_propagation(nn);
    // Return the final (output) layer
    return nn->neuron[nn->depth - 1];
}

// Loads a neural‐net model from disk (first line = quantized‐flag).
// If the first line is 0 → nn->quantized=false. If 1 → nn->quantized=true.
nn_t *nn_load_model(char *path)
{
    FILE *file = fopen(path, "r");
    if (file == NULL)
        return NULL;

    nn_t *nn = nn_init();
    if (nn == NULL) {
        fclose(file);
        return NULL;
    }

    // 1) Read quantized flag (0 or 1)
    int quant_flag = 0;
    if (fscanf(file, "%d\n", &quant_flag) != 1) {
        fclose(file);
        nn_free(nn);
        return NULL;
    }
    nn->quantized = (quant_flag != 0);

    int depth;
    // 2) Read depth
    if (fscanf(file, "%d\n", &depth) != 1) {
        fclose(file);
        nn_free(nn);
        return NULL;
    }

    // Read per‐layer (width, activation) and add layers
    for (int i = 0; i < depth; i++) {
        int width, activation;
        if (fscanf(file, "%d %d\n", &width, &activation) != 2) {
            fclose(file);
            nn_free(nn);
            return NULL;
        }
        if (nn_add_layer(nn, width, activation) != 0) {
            fclose(file);
            nn_free(nn);
            return NULL;
        }
    }

    // Read weights & biases for layers 1..depth-1
    for (int layer = 1; layer < nn->depth; layer++) {
        if (nn->quantized) {
            if (fscanf(file, "%f\n", &nn->bias_scale[layer]) != 1) {
                fclose(file);
                nn_free(nn);
                return NULL;
            }
        } else {
            float dummy;
            fscanf(file, "%f\n", &dummy);
        }
        for (int i = 0; i < (int)nn->width[layer]; i++) {
            if (nn->quantized) {
                if (fscanf(file, "%f\n", &nn->weight_scale[layer][i]) != 1) {
                    fclose(file);
                    nn_free(nn);
                    return NULL;
                }
            } else {
                float dummy;
                fscanf(file, "%f\n", &dummy);
            }
            for (int j = 0; j < (int)nn->width[layer - 1]; j++) {
                if (fscanf(file, "%f\n", &nn->weight[layer][i][j]) != 1) {
                    fclose(file);
                    nn_free(nn);
                    return NULL;
                }
            }
            if (fscanf(file, "%f\n", &nn->bias[layer][i]) != 1) {
                fclose(file);
                nn_free(nn);
                return NULL;
            }
        }
    }

    fclose(file);
    return nn;
}

// Saves a neural‐net model to disk. First line = quantized‐flag (0 or 1).  
// Then: depth, each layer’s width/activation, then weights & biases as before.
int nn_save_model(nn_t *nn, char *path)
{
    FILE *file = fopen(path, "w");
    if (file == NULL)
        return 1;

    // 1) Write quantized flag: 0 = floating, 1 = fixed‐point
    fprintf(file, "%d\n", nn->quantized ? 1 : 0);
    // 2) Write depth
    fprintf(file, "%" PRId32 "\n", nn->depth);
    // Write each layer's (width, activation)
    for (int i = 0; i < (int)nn->depth; i++) {
        fprintf(file, "%" PRId32 " %d\n", nn->width[i], nn->activation[i]);
    }

    // Write weights & biases
    for (int layer = 1; layer < (int)nn->depth; layer++) {
        // Save bias scale for this layer's set of biases
        if (nn->quantized)
            fprintf(file, "%f\n", nn->bias_scale[layer]);
        else
            fprintf(file, "0\n");
        for (int i = 0; i < (int)nn->width[layer]; i++) {
            // Save weight scale for this neuron's set of weights
            if (nn->quantized)
                fprintf(file, "%f\n", nn->weight_scale[layer][i]);
            else
                fprintf(file, "0\n");
            for (int j = 0; j < (int)nn->width[layer - 1]; j++) {
                fprintf(file, "%f\n", nn->weight[layer][i][j]);
            }
            fprintf(file, "%f\n", nn->bias[layer][i]);
        }
    }

    fclose(file);
    return 0;
}

int nn_remove_neuron(nn_t *nn, int layer, int neuron_index)
{
    if (nn == NULL || layer <= 0 || layer >= (int)nn->depth || neuron_index < 0 || neuron_index >= (int)nn->width[layer]) {
        return 1; // Invalid parameters
    }

    // Shift neuron, preact, loss arrays in this layer
    memmove(&nn->neuron[layer][neuron_index],
            &nn->neuron[layer][neuron_index + 1],
            sizeof(float) * (nn->width[layer] - neuron_index - 1));
    memmove(&nn->preact[layer][neuron_index],
            &nn->preact[layer][neuron_index + 1],
            sizeof(float) * (nn->width[layer] - neuron_index - 1));
    memmove(&nn->loss[layer][neuron_index],
            &nn->loss[layer][neuron_index + 1],
            sizeof(float) * (nn->width[layer] - neuron_index - 1));

    // Free the weight & weight_adj arrays belonging to the removed neuron
    free(nn->weight[layer][neuron_index]);
    free(nn->weight_adj[layer][neuron_index]);

    // Shift the pointers in this layer's weight & weight_adj arrays
    memmove(&nn->weight[layer][neuron_index],
            &nn->weight[layer][neuron_index + 1],
            sizeof(float *) * (nn->width[layer] - neuron_index - 1));
    memmove(&nn->weight_adj[layer][neuron_index],
            &nn->weight_adj[layer][neuron_index + 1],
            sizeof(float *) * (nn->width[layer] - neuron_index - 1));

    // Shrink the arrays for this layer
    nn->neuron[layer]     = (float *)realloc(nn->neuron[layer],     sizeof(float) * (nn->width[layer] - 1));
    nn->preact[layer]     = (float *)realloc(nn->preact[layer],     sizeof(float) * (nn->width[layer] - 1));
    nn->loss[layer]       = (float *)realloc(nn->loss[layer],       sizeof(float) * (nn->width[layer] - 1));
    nn->weight[layer]     = (float **)realloc(nn->weight[layer],     sizeof(float *) * (nn->width[layer] - 1));
    nn->weight_adj[layer] = (float **)realloc(nn->weight_adj[layer], sizeof(float *) * (nn->width[layer] - 1));

    // Update next layer's weights to remove the input connection from this neuron
    if (layer + 1 < (int)nn->depth) {
        for (int i = 0; i < (int)nn->width[layer + 1]; i++) {
            // Shift left the weights and weight_adj for next layer
            memmove(&nn->weight[layer + 1][i][neuron_index],
                    &nn->weight[layer + 1][i][neuron_index + 1],
                    sizeof(float) * (nn->width[layer] - neuron_index - 1));
            memmove(&nn->weight_adj[layer + 1][i][neuron_index],
                    &nn->weight_adj[layer + 1][i][neuron_index + 1],
                    sizeof(float) * (nn->width[layer] - neuron_index - 1));

            // Shrink each row in next layer
            nn->weight[layer + 1][i]     = (float *)realloc(nn->weight[layer + 1][i],     sizeof(float) * (nn->width[layer] - 1));
            nn->weight_adj[layer + 1][i] = (float *)realloc(nn->weight_adj[layer + 1][i], sizeof(float) * (nn->width[layer] - 1));
        }
    }

    // Finally, decrement the width of this layer
    nn->width[layer] -= 1;
    return 0;
}

// Returns the total weight associated with a given neuron, defined as the sum of the absolute values of both:
// Input weights (weights feeding into the neuron from the previous layer)
// Output weights (weights going out from the neuron to the next layer)
float nn_get_total_neuron_weight(nn_t *nn, int layer, int neuron_index)
{
    if (nn == NULL
     || layer <= 0
     || layer >= (int)nn->depth
     || neuron_index < 0
     || neuron_index >= (int)nn->width[layer]) {
        return 0.0f;
    }

    float total = 0.0f;

    // Sum absolute values of input weights (previous layer → this neuron)
    for (int i = 0; i < (int)nn->width[layer - 1]; i++) {
        total += fabsf(nn->weight[layer][neuron_index][i]);
    }

    // Sum absolute values of output weights (this neuron → next layer)
    if (layer + 1 < (int)nn->depth) {
        for (int i = 0; i < (int)nn->width[layer + 1]; i++) {
            total += fabsf(nn->weight[layer + 1][i][neuron_index]);
        }
    }

    return total;
}

bool nn_prune_lightest_neuron(nn_t *nn)
{
    if (nn == NULL || nn->depth < 2) {
	// Invalid or uninitialized network
        return false;
    }

    int lightest_layer = -1;
    int lightest_index = -1;
    float min_weight   = FLT_MAX;

    // Search all hidden layers (1..depth-2)
    for (int layer = 1; layer < (int)nn->depth - 1; layer++) {
        for (int neuron = 0; neuron < (int)nn->width[layer]; neuron++) {
            float tot = nn_get_total_neuron_weight(nn, layer, neuron);
            if (tot < min_weight) {
                min_weight = tot;
                lightest_layer = layer;
                lightest_index = neuron;
            }
        }
    }

    if (lightest_layer < 0) {
        return false;
    }

    // Remove that neuron
    nn_remove_neuron(nn, lightest_layer, lightest_index);
    return true;
}

void nn_pool2d(char *src, char *dest, int filter_size, int stride, pooling_type_t pooling_type, int x_in_size, int y_in_size, int *x_out_size, int *y_out_size)
{
    uint32_t pool_value;
    uint32_t pool_value_temp;

    *x_out_size = ((x_in_size - filter_size) / stride) + 1;
    *y_out_size = ((y_in_size - filter_size) / stride) + 1;

    // Assume src and dest are RGBA (4 channels)
    for (int z = 0; z < 4; z++) {
        for (int y_out = 0; y_out < *y_out_size; y_out++) {
            for (int x_out = 0; x_out < *x_out_size; x_out++) {
                switch (pooling_type) {
                    case POOLING_TYPE_MIN:
                        pool_value = 255;
                        for (int fy = 0; fy < filter_size; fy++) {
                            for (int fx = 0; fx < filter_size; fx++) {
                                pool_value_temp = (uint8_t)*(src + z + ((x_out + fx) * stride) * 4 + ((y_out + fy) * stride) * 4 * x_in_size);
                                if (pool_value_temp < pool_value) {
                                    pool_value = pool_value_temp;
                                }
                            }
                        }
                        break;

                    case POOLING_TYPE_MAX:
                        pool_value = 0;
                        for (int fy = 0; fy < filter_size; fy++) {
                            for (int fx = 0; fx < filter_size; fx++) {
                                pool_value_temp = (uint8_t)*(src + z + ((x_out + fx) * stride) * 4 + ((y_out + fy) * stride) * 4 * x_in_size);
                                if (pool_value_temp > pool_value) {
                                    pool_value = pool_value_temp;
                                }
                            }
                        }
                        break;

                    case POOLING_TYPE_AVG:
                        pool_value = 0;
                        for (int fy = 0; fy < filter_size; fy++) {
                            for (int fx = 0; fx < filter_size; fx++) {
                                pool_value += (uint8_t)*(src + z + ((x_out + fx) * stride) * 4 + ((y_out + fy) * stride) * 4 * x_in_size);
                            }
                        }
                        pool_value /= (filter_size * filter_size);
                        break;

                    case POOLING_TYPE_NONE:
                    default:
                        pool_value = 0;
                        break;
                }
                *(dest + z + (x_out * 4) + (y_out * (*x_out_size) * 4)) = (char)pool_value;
            }
        }
    }
}

void nn_conv2d(char *src, char *dest, int8_t *kernel, int kernel_size, int stride, activation_function_type_t activation_function_type, int x_in_size, int y_in_size, int *x_out_size, int *y_out_size)
{
    uint32_t kernel_value;

    *x_out_size = ((x_in_size - kernel_size) / stride) + 1;
    *y_out_size = ((y_in_size - kernel_size) / stride) + 1;

    // Assume RGBA—4 channels
    for (int z = 0; z < 4; z++) {
        for (int y_out = 0; y_out < *y_out_size; y_out++) {
            for (int x_out = 0; x_out < *x_out_size; x_out++) {
                switch (activation_function_type) {
                    case ACTIVATION_FUNCTION_TYPE_LINEAR:
                        kernel_value = 0;
                        for (int ky = 0; ky < kernel_size; ky++) {
                            for (int kx = 0; kx < kernel_size; kx++) {
                                uint8_t pixel = (uint8_t)*(src + z + ((x_out + kx) * stride) * 4 + ((y_out + ky) * stride) * 4 * x_in_size);
                                int8_t w = *(kernel + kx + ky * kernel_size);
                                kernel_value += (uint32_t)(pixel * w);
                            }
                        }
                        kernel_value /= (kernel_size * kernel_size);
                        break;

                    case ACTIVATION_FUNCTION_TYPE_NONE:
                    default:
                        kernel_value = 0;
                        break;
                }

                if (z == 3) {
                    // Alpha channel → max
                    *(dest + z + (x_out * 4) + (y_out * (*x_out_size) * 4)) = 255;
                } else {
                    *(dest + z + (x_out * 4) + (y_out * (*x_out_size) * 4)) = (char)kernel_value;
                }
            }
        }
    }
}
