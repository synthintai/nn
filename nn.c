/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
  const float s[] = {0.0f,      0.000045f, 0.000123f, 0.000335f, 0.000911f,
                     0.002473f, 0.006693f, 0.017986f, 0.047426f, 0.119203f,
                     0.268941f, 0.500000f, 0.731059f, 0.880797f, 0.952574f,
                     0.982014f, 0.993307f, 0.997527f, 0.999089f, 0.999665f,
                     0.999877f, 0.999955f, 1.0f};
  // Derivative of the sigmoid
  const float ds[] = {0.0f,      0.000045f, 0.000123f, 0.000335f, 0.000910f,
                      0.002467f, 0.006648f, 0.017663f, 0.045177f, 0.104994f,
                      0.196612f, 0.250000f, 0.196612f, 0.104994f, 0.045177f,
                      0.017663f, 0.006648f, 0.002466f, 0.000910f, 0.000335f,
                      0.000123f, 0.000045f, 0.0f};
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
    activation_function_tanh_fast};

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
  return ((uint32_t)NN_VERSION_MAJOR << 24) |
         ((uint32_t)NN_VERSION_MINOR << 16) |
         ((uint32_t)NN_VERSION_PATCH << 8) |
         (uint32_t)NN_VERSION_BUILD;
}

nn_t *nn_init(void)
{
  nn_t *nn = (nn_t *)malloc(sizeof(nn_t));
  if (nn == NULL)
    return NULL;
  // Mark as non‐quantized by default
  nn->quantized = false;
  // Populate version from macros
  nn->version_major = NN_VERSION_MAJOR;
  nn->version_minor = NN_VERSION_MINOR;
  nn->version_patch = NN_VERSION_PATCH;
  nn->version_build = NN_VERSION_BUILD;
  nn->depth = 0;
  nn->layer_type = NULL;
  nn->width = NULL;
  nn->activation = NULL;
  // Floats‐only pointers are NULL initially
  nn->neuron = NULL;
  nn->loss = NULL;
  nn->preact = NULL;
  nn->weight = NULL;
  nn->weight_adj = NULL;
  nn->bias = NULL;
  // Quantization‐related pointers start NULL
  nn->weight_quantized = NULL;
  nn->weight_scale = NULL;
  nn->bias_quantized = NULL;
  nn->bias_scale = NULL;
  return nn;
}

void nn_free(nn_t *nn)
{
  if (nn == NULL)
    return;
  // Free weight and bias related arrays layer by layer (float side)
  // There are no weights/biases for layer 0, so start from layer 1.
  if (!nn->quantized) {
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
  }
  // Free neuron, loss, preact arrays (no entry for layer 0 beyond input pointer)
  if (!nn->quantized) {
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      free(nn->neuron[layer]);
      free(nn->loss[layer]);
      free(nn->preact[layer]);
    }
    free(nn->weight);
    free(nn->weight_adj);
    free(nn->neuron);
    free(nn->loss);
    free(nn->preact);
    free(nn->layer_type);
    free(nn->width);
    free(nn->activation);
  }
  // Free quantized side arrays if allocated
  if (nn->quantized) {
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      int curr_w = nn->width[layer];
      for (int neuron = 0; neuron < curr_w; neuron++) {
        free(nn->weight_quantized[layer][neuron]);
      }
      free(nn->weight_quantized[layer]);
      free(nn->weight_scale[layer]);
      free(nn->bias_quantized[layer]);
    }
    free(nn->weight_quantized);
    free(nn->weight_scale);
    free(nn->bias_quantized);
    free(nn->bias_scale);
    // Also free the float side pointers that were allocated for prediction
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      free(nn->neuron[layer]);
      free(nn->loss[layer]);
      free(nn->preact[layer]);
    }
    free(nn->neuron);
    free(nn->loss);
    free(nn->preact);
    free(nn->layer_type);
    free(nn->width);
    free(nn->activation);
  }
  free(nn);
}

int nn_add_layer(nn_t *nn, layer_type_t layer_type, int width, int activation)
{
  // Increase depth by one
  nn->depth++;
  // Reallocate the layer_type array
  nn->layer_type = (uint8_t *)realloc(nn->layer_type, nn->depth * sizeof(*nn->layer_type));
  if (nn->layer_type == NULL)
    return 1;
  nn->layer_type[nn->depth - 1] = (uint8_t)layer_type;
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
  // Reallocate neuron, loss, and preact arrays
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
  // Reallocate top level weight, weight_adj, and bias arrays
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
    // Allocate per-neuron pointers in this new layer
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
      // Allocate the weight vector for this neuron
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

// Trains a nn with a given input and target output at a specified learning
// rate. The rate (or step size) controls how far in the search space to move
// against the gradient in each iteration of the algorithm. This function
// assumes quantized==false Returns the total error between the target and the
// output of the neural network.
float nn_train(nn_t *nn, float *inputs, float *targets, float rate)
{
  float sum;
  int i, j, k;

  if (nn->quantized) {
    // Cannot train a quantized network, so convert to a floating point model first.
    nn_dequantize(nn);
  }
  nn->neuron[0] = inputs;
  forward_propagation(nn);
  // Perform back propagation using gradient descent, which is an optimization
  // algorithm that follows the negative gradient of the objective function to
  // find the minimum of the function. Start at the output layer, and work
  // backward toward the input layer, adjusting weights along the way. Calculate
  // the error aka loss aka delta at the output.
  // Compute output layer loss
  i = (int)nn->depth - 1;
  for (j = 0; j < (int)nn->width[i]; j++) {
    nn->loss[i][j] = error_derivative(targets[j], nn->neuron[i][j]);
  }
  // Backpropagate loss into earlier layers
  for (i = nn->depth - 2; i > 0; i--) {
    for (j = 0; j < (int)nn->width[i]; j++) {
      sum = 0.0f;
      for (k = 0; k < (int)nn->width[i + 1]; k++) {
        // Apply the derivative of the activation function for the next layer’s neurons
        sum += nn->loss[i + 1][k] * activation_function[nn->activation[i + 1]](nn->preact[i + 1][k], true) * nn->weight[i + 1][k][j];
      }
      // The chain rule dictates that we should multiply the summed loss by the
      // derivative of the activation at the current neuron, not only during
      // weight updates, but immediately when calculating loss[i][j].
      nn->loss[i][j] = sum * activation_function[nn->activation[i]](nn->preact[i][j], true);
    }
  }
  // Update biases (gradient descent step)
  for (i = 1; i < (int)nn->depth; i++) {
    for (j = 0; j < (int)nn->width[i]; j++) {
      nn->bias[i][j] += nn->loss[i][j] * rate;
    }
  }
  // Calculate the weight adjustments. Note that their update is delayed until
  // after full backprop traversal. The weights cannot be updated while
  // back-propagating, because back propagating each layer depends on the next
  // layer's weights. So we save the weight adjustments in a temporary array and
  // apply them all at once later.
  // Compute weight adjustments (store in weight_adj)
  for (i = (int)nn->depth - 1; i > 0; i--) {
    for (j = 0; j < (int)nn->width[i]; j++) {
      for (k = 0; k < (int)nn->width[i - 1]; k++) {
        nn->weight_adj[i][j][k] = nn->loss[i][j] * nn->neuron[i - 1][k];
      }
    }
  }
  // Apply weight adjustments
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

// Returns an output prediction given an input.
float *nn_predict(nn_t *nn, float *inputs)
{
  nn->neuron[0] = inputs;
  forward_propagation(nn);
  // Return the output layer
  return nn->neuron[nn->depth - 1];
}

// Loads a neural net model from a file.
nn_t *nn_load_model_ascii(const char *path)
{
  FILE *file = fopen(path, "r");
  if (!file)
    return NULL;
  nn_t *nn = nn_init();
  if (!nn) {
    fclose(file);
    return NULL;
  }
  // Read quantized flag
  int quant_flag = 0;
  if (fscanf(file, "%d\n", &quant_flag) != 1) {
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  nn->quantized = (quant_flag != 0);
  // Read model version, use %hhu since version_* are uint8_t.
  if (fscanf(file, "%hhu %hhu %hhu %hhu\n", &nn->version_major, &nn->version_minor, &nn->version_patch, &nn->version_build) != 4) {
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  // Read depth
  int depth = 0;
  if (fscanf(file, "%d\n", &depth) != 1) {
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  // Call nn_add_layer for each (layer_type, width, activation)
  for (int i = 0; i < depth; i++) {
    int layer_type, w, act;
    if (fscanf(file, "%d %d %d\n", &layer_type, &w, &act) != 3) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
    if (nn_add_layer(nn, layer_type, w, act) != 0) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
  }
  // Allocate or reallocate neuron/loss/preact pointers for both float and quantized cases:
  nn->neuron = (float **)realloc(nn->neuron, nn->depth * sizeof(float *));
  nn->loss = (float **)realloc(nn->loss, nn->depth * sizeof(float *));
  nn->preact = (float **)realloc(nn->preact, nn->depth * sizeof(float *));
  if (!nn->neuron || !nn->loss || !nn->preact) {
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  // For layer 0 we do not allocate an array; layer 0's neuron pointer is set to inputs in nn_predict.
  for (int layer = 1; layer < nn->depth; layer++) {
    nn->neuron[layer] = (float *)malloc(nn->width[layer] * sizeof(float));
    nn->loss[layer] = (float *)malloc(nn->width[layer] * sizeof(float));
    nn->preact[layer] = (float *)malloc(nn->width[layer] * sizeof(float));
    if (!nn->neuron[layer] || !nn->loss[layer] || !nn->preact[layer]) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
  }
  // If float mode, read floats into nn->weight / nn->bias
  if (!nn->quantized) {
    for (int layer = 1; layer < nn->depth; layer++) {
      float dummy_scale;
      // Skip bias_scale (0)
      if (fscanf(file, "%f\n", &dummy_scale) != 1) {
        goto cleanup_float;
      }
      for (int i = 0; i < (int)nn->width[layer]; i++) {
        // Skip weight_scale (0)
        if (fscanf(file, "%f\n", &dummy_scale) != 1) {
          goto cleanup_float;
        }
        // Read float weights
        for (int j = 0; j < (int)nn->width[layer - 1]; j++) {
          if (fscanf(file, "%f\n", &nn->weight[layer][i][j]) != 1) {
            goto cleanup_float;
          }
        }
        // Read float bias
        if (fscanf(file, "%f\n", &nn->bias[layer][i]) != 1) {
          goto cleanup_float;
        }
      }
    }
    fclose(file);
    return nn;
  cleanup_float:
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  // Otherwise: quantized == true
  // Free all float‐side allocations made by nn_add_layer (weight, weight_adj, bias)
  for (int layer = 1; layer < nn->depth; layer++) {
    // Free per-neuron weight and weight_adj
    for (int i = 0; i < (int)nn->width[layer]; i++) {
      free(nn->weight[layer][i]);
      free(nn->weight_adj[layer][i]);
    }
    free(nn->weight[layer]);
    free(nn->weight_adj[layer]);
    free(nn->bias[layer]);
  }
  free(nn->weight);
  free(nn->weight_adj);
  free(nn->bias);
  // Note: we keep nn->neuron/loss/preact (allocated above) for use in nn_predict
  // Null out float-side pointers that won't be used for weights/biases
  nn->weight = NULL;
  nn->weight_adj = NULL;
  nn->bias = NULL;
  // weight_scale and bias_scale will be replaced by quantized arrays
  free(nn->weight_scale);
  free(nn->bias_scale);
  nn->weight_scale = NULL;
  nn->bias_scale = NULL;
  // Allocate top-level arrays for quantized model
  nn->weight_quantized = (int8_t ***)malloc(sizeof(int8_t **) * nn->depth);
  nn->weight_scale = (float **)malloc(sizeof(float *) * nn->depth);
  nn->bias_quantized = (int8_t **)malloc(sizeof(int8_t *) * nn->depth);
  nn->bias_scale = (float *)malloc(sizeof(float) * nn->depth);
  if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized ||
      !nn->bias_scale) {
    goto cleanup_quant_top;
  }
  // Initialize layer 0 entries
  nn->weight_quantized[0] = NULL;
  nn->weight_scale[0] = NULL;
  nn->bias_quantized[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  // Read quantized data, layer by layer
  int layer = 0, neuron = 0;
  for (layer = 1; layer < nn->depth; layer++) {
    int prev_w = nn->width[layer - 1];
    int curr_w = nn->width[layer];
    // Allocate per-layer arrays
    nn->weight_quantized[layer] = (int8_t **)malloc(sizeof(int8_t *) * curr_w);
    nn->weight_scale[layer] = (float *)malloc(sizeof(float) * curr_w);
    nn->bias_quantized[layer] = (int8_t *)malloc(sizeof(int8_t) * curr_w);
    if (!nn->weight_quantized[layer] || !nn->weight_scale[layer] || !nn->bias_quantized[layer]) {
      goto cleanup_quant_per_layer;
    }
    // For each neuron, read weight_scale + quantized weights
    for (neuron = 0; neuron < curr_w; neuron++) {
      if (fscanf(file, "%f\n", &nn->weight_scale[layer][neuron]) != 1) {
        goto cleanup_quant_data;
      }
      nn->weight_quantized[layer][neuron] = (int8_t *)malloc(sizeof(int8_t) * prev_w);
      if (!nn->weight_quantized[layer][neuron]) {
        goto cleanup_quant_data;
      }
      for (int w = 0; w < prev_w; w++) {
        int int_w;
        if (fscanf(file, "%d\n", &int_w) != 1) {
          goto cleanup_quant_data;
        }
        nn->weight_quantized[layer][neuron][w] = (int8_t)int_w;
      }
    }
    // Read bias_scale[layer]
    if (fscanf(file, "%f\n", &nn->bias_scale[layer]) != 1) {
      goto cleanup_quant_data;
    }
    // Read each neuron's quantized bias
    for (neuron = 0; neuron < curr_w; neuron++) {
      int int_b;
      if (fscanf(file, "%d\n", &int_b) != 1) {
        goto cleanup_quant_data;
      }
      nn->bias_quantized[layer][neuron] = (int8_t)int_b;
    }
  }
  fclose(file);
  return nn;
  // If we failed before allocating the top-level quantized arrays:
cleanup_quant_top:
  if (nn->weight_quantized)
    free(nn->weight_quantized);
  if (nn->weight_scale)
    free(nn->weight_scale);
  if (nn->bias_quantized)
    free(nn->bias_quantized);
  if (nn->bias_scale)
    free(nn->bias_scale);
  fclose(file);
  nn_free(nn);
  return NULL;
  // If we allocated some per-layer arrays for quantized, but then failed in that layer:
cleanup_quant_per_layer:
  // Free everything up to (layer−1):
  for (int L = 1; L < layer; L++) {
    int cw = nn->width[L];
    for (int n = 0; n < cw; n++) {
      free(nn->weight_quantized[L][n]);
    }
    free(nn->weight_quantized[L]);
    free(nn->weight_scale[L]);
    free(nn->bias_quantized[L]);
  }
  // Also free this layer's 'shell' arrays:
  if (nn->weight_quantized[layer])
    free(nn->weight_quantized[layer]);
  if (nn->weight_scale[layer])
    free(nn->weight_scale[layer]);
  if (nn->bias_quantized[layer])
    free(nn->bias_quantized[layer]);
  goto cleanup_quant_top;
  // If we allocated some int8 rows for the current layer but then failed reading data:
cleanup_quant_data:
  // In layer L, some neurons [0..neuron] have allocated int8 arrays, free them:
  for (int n = 0; n <= neuron; n++) {
    free(nn->weight_quantized[layer][n]);
  }
  // Free that layer's pointer block and scales:
  free(nn->weight_quantized[layer]);
  free(nn->weight_scale[layer]);
  free(nn->bias_quantized[layer]);
  // Free earlier layers too:
  for (int L = 1; L < layer; L++) {
    int cw = nn->width[L];
    for (int n = 0; n < cw; n++) {
      free(nn->weight_quantized[L][n]);
    }
    free(nn->weight_quantized[L]);
    free(nn->weight_scale[L]);
    free(nn->bias_quantized[L]);
  }
  goto cleanup_quant_top;
}

// Loads a neural-net model from a raw binary file.
nn_t *nn_load_model_binary(const char *path)
{
  FILE *file = fopen(path, "rb");
  if (!file)
    return NULL;
  nn_t *nn = nn_init();
  if (!nn) {
    fclose(file);
    return NULL;
  }
  // Quantized flag
  uint8_t qflag;
  if (fread(&qflag, sizeof(qflag), 1, file) != 1)
    goto error;
  nn->quantized = (qflag != 0);
  // Model version
  if (fread(&nn->version_major, sizeof(nn->version_major), 1, file) != 1)
    goto error;
  if (fread(&nn->version_minor, sizeof(nn->version_minor), 1, file) != 1)
    goto error;
  if (fread(&nn->version_patch, sizeof(nn->version_patch), 1, file) != 1)
    goto error;
  if (fread(&nn->version_build, sizeof(nn->version_build), 1, file) != 1)
    goto error;
  // Depth
  uint32_t depth;
  if (fread(&depth, sizeof(depth), 1, file) != 1)
    goto error;
  // Read each layer's width, layer type, and activation and call nn_add_layer()
  for (uint32_t i = 0; i < depth; i++) {
    layer_type_t layer_type;
    uint32_t w;
    uint8_t a;
    if (fread(&layer_type, sizeof(layer_type), 1, file) != 1)
      goto error;
    if (fread(&w, sizeof(w), 1, file) != 1)
      goto error;
    if (fread(&a, sizeof(a), 1, file) != 1)
      goto error;
    if (nn_add_layer(nn, layer_type, (int)w, (int)a) != 0)
      goto cleanup;
  }
  // Allocate neuron/loss/preact arrays
  nn->neuron = realloc(nn->neuron, depth * sizeof(float *));
  nn->loss = realloc(nn->loss, depth * sizeof(float *));
  nn->preact = realloc(nn->preact, depth * sizeof(float *));
  if (!nn->neuron || !nn->loss || !nn->preact)
    goto cleanup;
  for (int L = 1; L < (int)depth; L++) {
    nn->neuron[L] = malloc(nn->width[L] * sizeof(float));
    nn->loss[L] = malloc(nn->width[L] * sizeof(float));
    nn->preact[L] = malloc(nn->width[L] * sizeof(float));
    if (!nn->neuron[L] || !nn->loss[L] || !nn->preact[L])
      goto cleanup;
  }
  // Read weights & biases
  if (!nn->quantized) {
    // Float-mode: read dummy scales + real floats
    for (int L = 1; L < (int)depth; L++) {
      uint32_t curr = nn->width[L], prev = nn->width[L - 1];
      float dummy;
      // bias_scale placeholder
      if (fread(&dummy, sizeof(dummy), 1, file) != 1)
        goto cleanup;
      for (uint32_t i = 0; i < curr; i++) {
        // weight_scale placeholder
        if (fread(&dummy, sizeof(dummy), 1, file) != 1)
          goto cleanup;
        // weights
        if (fread(nn->weight[L][i], sizeof(float), prev, file) != prev)
          goto cleanup;
        // bias
        if (fread(&nn->bias[L][i], sizeof(float), 1, file) != 1)
          goto cleanup;
      }
    }
  } else {
    // Quantized-mode: free float-side, allocate quantized arrays
    for (int L = 1; L < (int)depth; L++) {
      for (int i = 0; i < (int)nn->width[L]; i++) {
        free(nn->weight[L][i]);
        free(nn->weight_adj[L][i]);
      }
      free(nn->weight[L]);
      free(nn->weight_adj[L]);
      free(nn->bias[L]);
    }
    free(nn->weight);
    nn->weight = NULL;
    free(nn->weight_adj);
    nn->weight_adj = NULL;
    free(nn->bias);
    nn->bias = NULL;
    free(nn->weight_scale);
    nn->weight_scale = NULL;
    free(nn->bias_scale);
    nn->bias_scale = NULL;
    // Top-level quant arrays
    nn->weight_quantized = malloc(depth * sizeof(int8_t **));
    nn->weight_scale = malloc(depth * sizeof(float *));
    nn->bias_quantized = malloc(depth * sizeof(int8_t *));
    nn->bias_scale = malloc(depth * sizeof(float));
    if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized ||
        !nn->bias_scale)
      goto cleanup;
    // Layer 0
    nn->weight_quantized[0] = NULL;
    nn->weight_scale[0] = NULL;
    nn->bias_quantized[0] = NULL;
    nn->bias_scale[0] = 0.0f;
    // Read per-layer quant data
    for (int L = 1; L < (int)depth; L++) {
      uint32_t curr = nn->width[L], prev = nn->width[L - 1];
      // Allocate per-layer
      nn->weight_quantized[L] = malloc(curr * sizeof(int8_t *));
      nn->weight_scale[L] = malloc(curr * sizeof(float));
      nn->bias_quantized[L] = malloc(curr * sizeof(int8_t));
      if (!nn->weight_quantized[L] || !nn->weight_scale[L] ||
          !nn->bias_quantized[L])
        goto cleanup;
      // Read each neuron's weight_scale and weights
      for (uint32_t i = 0; i < curr; i++) {
        if (fread(&nn->weight_scale[L][i], sizeof(float), 1, file) != 1)
          goto cleanup;
        nn->weight_quantized[L][i] = malloc(prev * sizeof(int8_t));
        if (!nn->weight_quantized[L][i])
          goto cleanup;
        if (fread(nn->weight_quantized[L][i], sizeof(int8_t), prev, file) !=
            prev)
          goto cleanup;
      }
      // Read bias_scale[L]
      if (fread(&nn->bias_scale[L], sizeof(float), 1, file) != 1)
        goto cleanup;
      // Read quantized biases
      if (fread(nn->bias_quantized[L], sizeof(int8_t), curr, file) != curr)
        goto cleanup;
    }
  }
  fclose(file);
  return nn;
cleanup:
  fclose(file);
  nn_free(nn);
  return NULL;
error:
  fclose(file);
  nn_free(nn);
  return NULL;
}

// Saves a neural net model to a file.
int nn_save_model_ascii(nn_t *nn, const char *path)
{
  FILE *file = fopen(path, "w");
  if (file == NULL)
    return 1;
  // Write quantized flag: 0 = floating, 1 = fixed‐point
  fprintf(file, "%d\n", nn->quantized ? 1 : 0);
  // Write model version (major, minor, patch, build)
  fprintf(file, "%hhu %hhu %hhu %hhu\n", nn->version_major, nn->version_minor, nn->version_patch, nn->version_build);
  // Write depth
  fprintf(file, "%" PRId32 "\n", nn->depth);
  // Write each layer's width, layer type, and activation
  for (int i = 0; i < (int)nn->depth; i++) {
    fprintf(file, "%" PRId32 " %d %d\n", nn->layer_type[i], nn->width[i], nn->activation[i]);
  }
  // Write weights & biases
  if (!nn->quantized) {
    // Float mode: write weight_scale (0), weights, and bias
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      // bias_scale placeholder
      fprintf(file, "0\n");
      for (int i = 0; i < (int)nn->width[layer]; i++) {
        // weight_scale placeholder
        fprintf(file, "0\n");
        for (int j = 0; j < (int)nn->width[layer - 1]; j++) {
          fprintf(file, "%f\n", nn->weight[layer][i][j]);
        }
        fprintf(file, "%f\n", nn->bias[layer][i]);
      }
    }
  } else {
    // Quantized mode: write weight_scale, quantized weights, bias_scale, quantized bias
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      int prev_w = nn->width[layer - 1];
      int curr_w = nn->width[layer];
      for (int neuron = 0; neuron < curr_w; neuron++) {
        fprintf(file, "%f\n", nn->weight_scale[layer][neuron]);
        for (int w = 0; w < prev_w; w++) {
          fprintf(file, "%d\n", (int)nn->weight_quantized[layer][neuron][w]);
        }
      }
      fprintf(file, "%f\n", nn->bias_scale[layer]);
      for (int neuron = 0; neuron < curr_w; neuron++) {
        fprintf(file, "%d\n", (int)nn->bias_quantized[layer][neuron]);
      }
    }
  }
  fclose(file);
  return 0;
}

// Exports a neural-net model as raw binary.
int nn_save_model_binary(nn_t *nn, const char *path)
{
  FILE *file = fopen(path, "wb");
  if (!file)
    return 1;
  // Quantized flag
  uint8_t qflag = nn->quantized ? 1 : 0;
  fwrite(&qflag, sizeof(qflag), 1, file);
  // Model version (major, minor, patch, build)
  fwrite(&nn->version_major, sizeof(nn->version_major), 1, file);
  fwrite(&nn->version_minor, sizeof(nn->version_minor), 1, file);
  fwrite(&nn->version_patch, sizeof(nn->version_patch), 1, file);
  fwrite(&nn->version_build, sizeof(nn->version_build), 1, file);
  // Depth
  uint32_t depth = nn->depth;
  fwrite(&depth, sizeof(depth), 1, file);
  // layer_type, width, and activation per layer
  for (uint32_t i = 0; i < depth; i++) {
    uint8_t layer_type = nn->layer_type[i];
    uint32_t w = nn->width[i];
    uint8_t a = nn->activation[i];
    fwrite(&layer_type, sizeof(layer_type), 1, file);
    fwrite(&w, sizeof(w), 1, file);
    fwrite(&a, sizeof(a), 1, file);
  }
  // Weights & biases
  if (!nn->quantized) {
    // Float mode: placeholders for scales and actual floats
    for (uint32_t L = 1; L < depth; L++) {
      float bias_scale = 0.0f;
      fwrite(&bias_scale, sizeof(bias_scale), 1, file);
      uint32_t curr = nn->width[L], prev = nn->width[L - 1];
      for (uint32_t i = 0; i < curr; i++) {
        float weight_scale = 0.0f;
        fwrite(&weight_scale, sizeof(weight_scale), 1, file);
        fwrite(nn->weight[L][i], sizeof(float), prev, file);
        fwrite(&nn->bias[L][i], sizeof(float), 1, file);
      }
    }
  } else {
    // Quantized mode: real scales and int8 quantized data
    for (uint32_t L = 1; L < depth; L++) {
      uint32_t curr = nn->width[L], prev = nn->width[L - 1];
      for (uint32_t i = 0; i < curr; i++) {
        // Per-neuron weight scale
        fwrite(&nn->weight_scale[L][i], sizeof(float), 1, file);
        // Quantized weights
        fwrite(nn->weight_quantized[L][i], sizeof(int8_t), prev, file);
      }
      // Bias scale (one per layer)
      fwrite(&nn->bias_scale[L], sizeof(float), 1, file);
      // Quantized biases
      fwrite(nn->bias_quantized[L], sizeof(int8_t), curr, file);
    }
  }
  fclose(file);
  return 0;
}

int nn_remove_neuron(nn_t *nn, int layer, int neuron_index)
{
  if (nn == NULL || layer <= 0 || layer >= (int)nn->depth || neuron_index < 0 || neuron_index >= (int)nn->width[layer]) {
    return 1;
  }
  int old_width = nn->width[layer];
  // Shift out neuron / preact / loss in this layer
  memmove(&nn->neuron[layer][neuron_index], &nn->neuron[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  memmove(&nn->preact[layer][neuron_index], &nn->preact[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  memmove(&nn->loss[layer][neuron_index], &nn->loss[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  // Free exactly one removed‐neuron row (weights and weight_adj)
  if (nn->quantized) {
    // Free the int8_t row of input weights for this neuron
    free(nn->weight_quantized[layer][neuron_index]);
    // Do NOT free bias_quantized[layer][neuron_index] here!
    // Instead, we will shift the entire bias_quantized[layer] array and then realloc it below.
  } else {
    free(nn->weight[layer][neuron_index]);
    free(nn->weight_adj[layer][neuron_index]);
  }
  // Shift pointers / elements within this layer
  if (nn->quantized) {
    // Shift pointer array for weight_quantized[layer]
    memmove(&nn->weight_quantized[layer][neuron_index], &nn->weight_quantized[layer][neuron_index + 1], sizeof(int8_t *) * (old_width - neuron_index - 1));
    // Shift the single‐byte biases in bias_quantized[layer]
    memmove(&nn->bias_quantized[layer][neuron_index], &nn->bias_quantized[layer][neuron_index + 1], sizeof(int8_t) * (old_width - neuron_index - 1));
    // Leave bias_scale[layer] alone (it’s a single float per layer).
    // The float‐side arrays (weight[layer], weight_adj[layer], bias[layer]) are NULL here, so we must NOT touch them in quantized mode.
  } else {
    // Shift float pointers for weight[layer] and weight_adj[layer]
    memmove(&nn->weight[layer][neuron_index], &nn->weight[layer][neuron_index + 1], sizeof(float *) * (old_width - neuron_index - 1));
    memmove(&nn->weight_adj[layer][neuron_index], &nn->weight_adj[layer][neuron_index + 1], sizeof(float *) * (old_width - neuron_index - 1));
    // Shift the float biases in bias[layer]
    memmove(&nn->bias[layer][neuron_index], &nn->bias[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  }
  // Realloc every array in this layer:
  if (nn->quantized) {
    // Shrink weight_quantized[layer] (pointer‐to‐pointer)
    nn->weight_quantized[layer] = (int8_t **)realloc(nn->weight_quantized[layer], sizeof(int8_t *) * (old_width - 1));
    // Shrink the 1‐D bias array bias_quantized[layer]:
    nn->bias_quantized[layer] = (int8_t *)realloc(nn->bias_quantized[layer], sizeof(int8_t) * (old_width - 1));
    // bias_scale[layer] remains a single float, so no realloc.
  } else {
    // Shrink weight[layer] pointer array
    nn->weight[layer] = (float **)realloc(nn->weight[layer], sizeof(float *) * (old_width - 1));
    // Shrink weight_adj[layer] pointer array
    nn->weight_adj[layer] = (float **)realloc(nn->weight_adj[layer], sizeof(float *) * (old_width - 1));
    // Shrink the float bias array bias[layer]
    nn->bias[layer] = (float *)realloc(nn->bias[layer], sizeof(float) * (old_width - 1));
  }
  // Update next layer's weights to remove the input connection from this neuron
  if (layer + 1 < (int)nn->depth) {
    int old_prev_width = old_width;
    int next_width = nn->width[layer + 1];
    for (int i = 0; i < next_width; i++) {
      if (nn->quantized) {
        // Shift out column "neuron_index" from each int8 row
        memmove(&nn->weight_quantized[layer + 1][i][neuron_index], &nn->weight_quantized[layer + 1][i][neuron_index + 1], sizeof(int8_t) * (old_prev_width - neuron_index - 1));
        // Now shrink that row to (old_prev_width - 1) bytes:
        nn->weight_quantized[layer + 1][i] = (int8_t *)realloc(nn->weight_quantized[layer + 1][i], sizeof(int8_t) * (old_prev_width - 1));
        // Do NOT touch any biases in layer+1.
      } else {
        // Shift out column "neuron_index" from each float row
        memmove(&nn->weight[layer + 1][i][neuron_index], &nn->weight[layer + 1][i][neuron_index + 1], sizeof(float) * (old_prev_width - neuron_index - 1));
        // Shrink that row to (old_prev_width - 1) floats
        nn->weight[layer + 1][i] = (float *)realloc(nn->weight[layer + 1][i], sizeof(float) * (old_prev_width - 1));
        // Also shift & shrink weight_adj[row]
        memmove(&nn->weight_adj[layer + 1][i][neuron_index], &nn->weight_adj[layer + 1][i][neuron_index + 1], sizeof(float) * (old_prev_width - neuron_index - 1));
        nn->weight_adj[layer + 1][i] = (float *)realloc(nn->weight_adj[layer + 1][i], sizeof(float) * (old_prev_width - 1));
        // Do NOT touch any biases in layer+1.
      }
    }
  }
  // Decrement the width of this layer
  nn->width[layer] = old_width - 1;
  return 0;
}

// Returns the total weight associated with a given neuron, defined as the sum
// of the absolute values of both: Input weights (weights feeding into the
// neuron from the previous layer) Output weights (weights going out from the
// neuron to the next layer)
float nn_get_total_neuron_weight(nn_t *nn, int layer, int neuron_index)
{
  if (nn == NULL || layer <= 0 || layer >= (int)nn->depth || neuron_index < 0 ||
      neuron_index >= (int)nn->width[layer]) {
    return 0.0f;
  }
  float total = 0.0f;
  // Sum absolute values of input weights (previous layer to this neuron)
  for (int i = 0; i < (int)nn->width[layer - 1]; i++) {
    if (nn->quantized) {
      // For quantized models, we use the quantized weights
      total += fabsf((float)nn->weight_quantized[layer][neuron_index][i] * nn->weight_scale[layer][neuron_index]);
    } else {
      // For float models, we use the float weights directly
      total += fabsf(nn->weight[layer][neuron_index][i]);
    }
  }
  // Sum absolute values of output weights (this neuron to next layer)
  if (layer + 1 < (int)nn->depth) {
    for (int i = 0; i < (int)nn->width[layer + 1]; i++) {
      if (nn->quantized) {
        // For quantized models, we use the quantized weights
        total += fabsf((float)nn->weight_quantized[layer + 1][i][neuron_index] * nn->weight_scale[layer + 1][i]);
      } else {
        // For float models, we use the float weights directly
        total += fabsf(nn->weight[layer + 1][i][neuron_index]);
      }
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
  float min_weight = FLT_MAX;
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

// In-place quantization of nn_t
int nn_quantize(nn_t *nn)
{
  if (!nn || nn->quantized) {
    // Nothing to do
    return -1;
  }
  const int depth = (int)nn->depth;
  // Mark the network as quantized
  nn->quantized = true;
  // Allocate quantization arrays in the union fields
  nn->weight_quantized = malloc(depth * sizeof(int8_t **));
  nn->weight_scale = malloc(depth * sizeof(float *));
  nn->bias_quantized = malloc(depth * sizeof(int8_t *));
  nn->bias_scale = malloc(depth * sizeof(float));
  if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized ||
      !nn->bias_scale) {
    return -1;
  }
  // Initialize layer 0 entries
  nn->weight_quantized[0] = NULL;
  nn->weight_scale[0] = NULL;
  nn->bias_quantized[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  // Quantize each layer > 1
  for (int L = 1; L < depth; L++) {
    int prev_w = nn->width[L - 1];
    int curr_w = nn->width[L];
    // Allocate per-layer arrays
    nn->weight_quantized[L] = malloc(curr_w * sizeof(int8_t *));
    nn->weight_scale[L] = malloc(curr_w * sizeof(float));
    nn->bias_quantized[L] = malloc(curr_w * sizeof(int8_t));
    if (!nn->weight_quantized[L] || !nn->weight_scale[L] || !nn->bias_quantized[L]) {
      return -1;
    }
    // Compute one bias_scale for this layer
    float min_b = nn->bias[L][0], max_b = min_b;
    for (int i = 1; i < curr_w; i++) {
      float v = nn->bias[L][i];
      if (v < min_b)
        min_b = v;
      if (v > max_b)
        max_b = v;
    }
    float layer_bias_scale = fmaxf(fabsf(min_b), fabsf(max_b)) / 127.0f;
    if (layer_bias_scale == 0.0f)
      layer_bias_scale = 1e-8f;
    nn->bias_scale[L] = layer_bias_scale;
    // For each neuron in this layer:
    for (int n = 0; n < curr_w; n++) {
      // Allocate the int8 weight-vector
      nn->weight_quantized[L][n] = malloc(prev_w * sizeof(int8_t));
      if (!nn->weight_quantized[L][n])
        return -1;
      // Compute per-neuron weight scale
      float min_w = nn->weight[L][n][0], max_w = min_w;
      for (int k = 1; k < prev_w; k++) {
        float w = nn->weight[L][n][k];
        if (w < min_w)
          min_w = w;
        if (w > max_w)
          max_w = w;
      }
      float neuron_scale = fmaxf(fabsf(min_w), fabsf(max_w)) / 127.0f;
      if (neuron_scale == 0.0f)
        neuron_scale = 1e-8f;
      nn->weight_scale[L][n] = neuron_scale;
      // Quantize each weight
      for (int k = 0; k < prev_w; k++) {
        float orig = nn->weight[L][n][k];
        int8_t q = (int8_t)lroundf(orig / neuron_scale);
        nn->weight_quantized[L][n][k] = (q > 127 ? 127 : (q < -128 ? -128 : q));
      }
      // Quantize the bias
      float borig = nn->bias[L][n];
      int8_t bq = (int8_t)lroundf(borig / layer_bias_scale);
      nn->bias_quantized[L][n] = (bq > 127 ? 127 : (bq < -128 ? -128 : bq));
    }
  }
  // Free all of the original float-side storage AFTER quantization
  for (int L = 1; L < depth; L++) {
    int wcount = (int)nn->width[L];
    for (int n = 0; n < wcount; n++) {
      free(nn->weight[L][n]);
      free(nn->weight_adj[L][n]);
    }
    free(nn->weight[L]);
    free(nn->weight_adj[L]);
    free(nn->bias[L]);
  }
  free(nn->weight);
  free(nn->weight_adj);
  free(nn->bias);
  return 0;
}

// Dequantize in-place: rebuild float weights/biases from the fixed-point model.
// Returns 0 on success, -1 on error.
int nn_dequantize(nn_t *nn)
{
  if (!nn || !nn->quantized) {
    return -1;
  }
  const int depth = (int)nn->depth;
  // Allocate top-level float pointers
  nn->weight = malloc(depth * sizeof(*nn->weight));
  nn->weight_adj = malloc(depth * sizeof(*nn->weight_adj));
  nn->bias = malloc(depth * sizeof(*nn->bias));
  if (!nn->weight || !nn->weight_adj || !nn->bias) {
    return -1;
  }
  // For each layer >=1, rebuild float weight, weight_adj, bias
  for (int L = 1; L < depth; L++) {
    int curr = (int)nn->width[L];
    int prev = (int)nn->width[L - 1];
    // Allocate per-layer pointer arrays
    nn->weight[L] = malloc(curr * sizeof(*nn->weight[L]));
    nn->weight_adj[L] = malloc(curr * sizeof(*nn->weight_adj[L]));
    nn->bias[L] = malloc(curr * sizeof(*nn->bias[L]));
    if (!nn->weight[L] || !nn->weight_adj[L] || !nn->bias[L]) {
      return -1;
    }
    // For each neuron, allocate its float row, fill from quantized
    for (int i = 0; i < curr; i++) {
      // Allocate float-weight row and float-weight_adj row
      nn->weight[L][i] = malloc(prev * sizeof(*nn->weight[L][i]));
      nn->weight_adj[L][i] = malloc(prev * sizeof(*nn->weight_adj[L][i]));
      if (!nn->weight[L][i] || !nn->weight_adj[L][i]) {
        return -1;
      }
      // Dequantize each weight
      float wscale = nn->weight_scale[L][i];
      for (int j = 0; j < prev; j++) {
        nn->weight[L][i][j] = nn->weight_quantized[L][i][j] * wscale;
        // weight_adj was never meaningful in quantized mode, zero it
        nn->weight_adj[L][i][j] = 0.0f;
      }
      // Dequantize bias
      nn->bias[L][i] = nn->bias_quantized[L][i] * nn->bias_scale[L];
      // Free the now-unused row of quantized weights
      free(nn->weight_quantized[L][i]);
    }
    // Free per-layer quant arrays
    free(nn->weight_quantized[L]);
    free(nn->weight_scale[L]);
    free(nn->bias_quantized[L]);
  }
  // Free top-level quant pointers
  free(nn->weight_quantized);
  free(nn->weight_scale);
  free(nn->bias_quantized);
  free(nn->bias_scale);
  // Mark as float mode
  nn->quantized = false;
  return 0;
}
