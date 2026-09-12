/*
 * Neural Network library
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ctype.h>
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

// First 4 bytes of every binary model file, so nn_load_model() can tell a
// binary model file apart from an ascii one by content rather than by
// trusting the file extension.
#define NN_BINARY_MAGIC_LEN 4
static const uint8_t NN_BINARY_MAGIC[NN_BINARY_MAGIC_LEN] = {'N', 'N', 'B', '1'};

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
  // For a < 0, elu(a) = expf(a) - 1, so elu'(a) = expf(a) (not elu(a) itself,
  // i.e. not expf(a) - 1).
  return (derivative ? expf(a) : expf(a) - 1);
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

// Gaussian Error Linear Unit (GELU) activation function
static float activation_function_gelu(float a, bool derivative)
{
  const float inv_sqrt2 = 0.70710678118654752440f;
  const float inv_sqrt2pi = 0.39894228040143267794f;
  float cdf = 0.5f * (1.0f + erff(a * inv_sqrt2));
  if (derivative) {
    float pdf = inv_sqrt2pi * expf(-0.5f * a * a);
    return cdf + a * pdf;
  }
  return a * cdf;
}

// Sigmoid Linear Unit (SiLU) activation function (aka Swish)
static float activation_function_silu(float a, bool derivative)
{
  float s = activation_function_sigmoid(a, false);
  if (derivative)
    return s + a * s * (1.0f - s);
  return a * s;
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
    activation_function_tanh_fast,
    activation_function_gelu,
    activation_function_silu};

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

// Quantization/dequantization treat each layer as a set of "rows" that share
// a single layer-wide bias scale: for a CNN layer a row is a kernel
// (out_channels * in_channels of them, row_len = kernel_size^2 wide), with
// bias_count = out_channels; for FC/OUTPUT layers a row is a neuron
// (width[L] of them, row_len = width[L-1] wide), with bias_count =
// width[L]. Pooling layers have no rows at all -- callers must check
// nn->layer_type[L] != LAYER_TYPE_POOL themselves before using this.
static void quantized_layer_shape(nn_t *nn, int L, int *rows, int *row_len, int *bias_count)
{
  if (nn->layer_type[L] == LAYER_TYPE_CNN) {
    cnn_t *c = nn->config[L];
    *rows = c->out_channels * c->in_channels;
    *row_len = c->kernel_size * c->kernel_size;
    *bias_count = c->out_channels;
  } else {
    *rows = (int)nn->width[L];
    *row_len = (int)nn->width[L - 1];
    *bias_count = (int)nn->width[L];
  }
}

// 2D pooling layer forward pass over the previous layer's (float) feature
// maps. Supports MIN/MAX ("winner take all", cached for backprop routing)
// and AVG (uniform reduction) pooling. Pooling has no weights/bias and no
// activation function of its own -- preact[layer] simply mirrors neuron[layer]
// so the generic activation-derivative machinery in nn_train() (with
// nn->activation[layer] set to ACTIVATION_FUNCTION_TYPE_LINEAR) is a no-op.
void nn_pool_forward(nn_t *nn, int layer)
{
  pool_t *pool = nn->config[layer];
  const int channels = pool->channels;
  const int psize = pool->pool_size;
  const int stride = pool->stride;
  const int x_out = pool->out_w;
  const int y_out = pool->out_h;
  const int plane_out = x_out * y_out;
  const int plane_in = pool->in_w * pool->in_h;
  // Sanity-check shapes
  if ((channels <= 0) || ((uint32_t)channels * plane_out != nn->width[layer])) {
    fprintf(stderr, "pool2d: inconsistent shape (channels=%d, plane_out=%d, width[%d]=%u)\n", channels, plane_out, layer, nn->width[layer]);
    return;
  }
  const float *in = nn->neuron[layer - 1];
  float *out = nn->neuron[layer];
  float *pre = nn->preact[layer];
  int *argmax = nn->pool_argmax[layer]; // NULL unless MIN/MAX pooling

  for (int c = 0; c < channels; ++c) {
    for (int oy = 0; oy < y_out; ++oy) {
      const int in_y = oy * stride;
      for (int ox = 0; ox < x_out; ++ox) {
        const int in_x = ox * stride;
        const int oidx = c * plane_out + oy * x_out + ox;
        float result = 0.0f;
        int winner = -1;
        switch (pool->pooling_type) {
          case POOLING_TYPE_MAX: {
            float best = -FLT_MAX;
            for (int ky = 0; ky < psize; ++ky) {
              for (int kx = 0; kx < psize; ++kx) {
                int in_idx = c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx);
                if (in[in_idx] > best) {
                  best = in[in_idx];
                  winner = in_idx;
                }
              }
            }
            result = best;
            break;
          }
          case POOLING_TYPE_MIN: {
            float best = FLT_MAX;
            for (int ky = 0; ky < psize; ++ky) {
              for (int kx = 0; kx < psize; ++kx) {
                int in_idx = c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx);
                if (in[in_idx] < best) {
                  best = in[in_idx];
                  winner = in_idx;
                }
              }
            }
            result = best;
            break;
          }
          case POOLING_TYPE_AVG: {
            float sum = 0.0f;
            for (int ky = 0; ky < psize; ++ky)
              for (int kx = 0; kx < psize; ++kx)
                sum += in[c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx)];
            result = sum / (float)(psize * psize);
            break;
          }
          case POOLING_TYPE_NONE:
          default:
            result = 0.0f;
            break;
        }
        out[oidx] = result;
        pre[oidx] = result;
        if (argmax)
          argmax[oidx] = winner;
      }
    }
  }
}

// Routes/distributes a pooling layer's loss (dE/d(pool output), already
// computed in nn->loss[layer]) back into grad_in, which must be zeroed by the
// caller and sized to the pooling layer's input width (i.e. the previous
// layer's width). MIN/MAX pooling route the full gradient to the single
// cached winning input position; AVG pooling distributes it evenly across
// every input position in the window.
static void nn_pool_backward(nn_t *nn, int layer, float *grad_in)
{
  pool_t *pool = nn->config[layer];
  const int channels = pool->channels;
  const int psize = pool->pool_size;
  const int stride = pool->stride;
  const int x_out = pool->out_w;
  const int y_out = pool->out_h;
  const int plane_out = x_out * y_out;
  const int plane_in = pool->in_w * pool->in_h;
  const float *loss = nn->loss[layer];
  const int *argmax = nn->pool_argmax[layer];

  for (int c = 0; c < channels; ++c) {
    for (int oy = 0; oy < y_out; ++oy) {
      const int in_y = oy * stride;
      for (int ox = 0; ox < x_out; ++ox) {
        const int in_x = ox * stride;
        const int oidx = c * plane_out + oy * x_out + ox;
        const float g = loss[oidx];
        switch (pool->pooling_type) {
          case POOLING_TYPE_MAX:
          case POOLING_TYPE_MIN:
            grad_in[argmax[oidx]] += g;
            break;
          case POOLING_TYPE_AVG: {
            const float share = g / (float)(psize * psize);
            for (int ky = 0; ky < psize; ++ky)
              for (int kx = 0; kx < psize; ++kx)
                grad_in[c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx)] += share;
            break;
          }
          case POOLING_TYPE_NONE:
          default:
            break;
        }
      }
    }
  }
}

static void forward_propagation(nn_t *nn)
{
  float sum;
  int i, j, k;

  // Calculate neuron values in each layer
  for (i = 1; i < (int)nn->depth; i++) {
    switch(nn->layer_type[i]) {
      case LAYER_TYPE_FC:
      case LAYER_TYPE_OUTPUT:
        // Fully Connected Layer. The quantized/float branch and the
        // per-neuron weight_scale multiply are hoisted out of the innermost
        // dot-product loop (checked/applied once per neuron rather than
        // once per weight) so the accumulation loop is a plain, easily
        // vectorized float multiply-add either way.
        {
          const int row_len = (int)nn->width[i - 1]; // flat weight buffer stride for this layer
          if (nn->quantized) {
            for (j = 0; j < (int)nn->width[i]; j++) {
              sum = 0.0f;
              const int8_t *wrow = nn->weight_quantized[i] + j * row_len;
              for (k = 0; k < row_len; k++) {
                sum += nn->neuron[i - 1][k] * (float)wrow[k];
              }
              sum = sum * nn->weight_scale[i][j] + (float)nn->bias_quantized[i][j] * nn->bias_scale[i];
              nn->neuron[i][j] = activation_function[nn->activation[i]](sum, false);
              nn->preact[i][j] = sum;
            }
          } else {
            for (j = 0; j < (int)nn->width[i]; j++) {
              sum = 0.0f;
              const float *wrow = nn->weight[i] + j * row_len;
              for (k = 0; k < row_len; k++) {
                sum += nn->neuron[i - 1][k] * wrow[k];
              }
              sum += nn->bias[i][j];
              nn->neuron[i][j] = activation_function[nn->activation[i]](sum, false);
              nn->preact[i][j] = sum;
            }
          }
        }
        break;
      case LAYER_TYPE_CNN:
        // Convolutional Neural Network Layer
        nn_conv2d(nn, i);
        break;
      case LAYER_TYPE_POOL:
        // Pooling Layer
        nn_pool_forward(nn, i);
        break;
      case LAYER_TYPE_LSTM:
        // Long Short-Term Memory Layer
        // TODO
        break;
      case LAYER_TYPE_GRU:
        // Gated Recurrent Unit Layer
        // TODO
        break;
      case LAYER_TYPE_RNN:
        // Recurrent Neural Network Layer
        // TODO
        break;
      case LAYER_TYPE_ATTENTION:
        // Attention Layer
        // TODO
        break;
      case LAYER_TYPE_TRANSFORMER:
        // Transformer Layer
        // TODO
        break;
      case LAYER_TYPE_INPUT:
      case LAYER_TYPE_NONE:
      default:
        // No operation for this layer
        break;
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
  // Mark as non-quantized by default
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
  nn->config = NULL;
  // Floats-only pointers are NULL initially
  nn->neuron = NULL;
  nn->loss = NULL;
  nn->preact = NULL;
  nn->weight = NULL;
  nn->weight_adj = NULL;
  nn->bias = NULL;
  // Quantization-related pointers start NULL
  nn->weight_quantized = NULL;
  nn->weight_scale = NULL;
  nn->bias_quantized = NULL;
  nn->bias_scale = NULL;
  nn->pool_argmax = NULL;
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
      // weight/weight_adj are each one flat buffer per layer (or NULL for
      // POOL); free(NULL) is a no-op so this is safe uniformly across
      // every layer type.
      free(nn->weight[layer]);
      free(nn->weight_adj[layer]);
      free(nn->bias[layer]);
      free(nn->weight_scale[layer]);
      free(nn->config[layer]);
      free(nn->neuron[layer]);
      free(nn->loss[layer]);
      free(nn->preact[layer]);
    }
    free(nn->weight);
    free(nn->weight_adj);
    free(nn->weight_scale);
    free(nn->bias);
    free(nn->bias_scale);
    free(nn->neuron);
    free(nn->loss);
    free(nn->preact);
    free(nn->layer_type);
    free(nn->width);
    free(nn->activation);
    free(nn->config);
  }
  // Free quantized side arrays if allocated
  if (nn->quantized) {
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      // weight_quantized is one flat buffer per layer (or NULL for POOL);
      // free(NULL) is a no-op so this is safe uniformly. Guard the whole
      // group on the top-level pointer, though: a load failure part-way
      // through reading a quantized file can leave weight_quantized (and
      // its siblings) already torn down and NULLed by the caller, in which
      // case there's nothing left here to index into.
      if (nn->weight_quantized) {
        free(nn->weight_quantized[layer]);
        free(nn->weight_scale[layer]);
        free(nn->bias_quantized[layer]);
      }
      free(nn->config[layer]);
      free(nn->neuron[layer]);
      free(nn->loss[layer]);
      free(nn->preact[layer]);
    }
    free(nn->weight_quantized);
    free(nn->weight_scale);
    free(nn->bias_quantized);
    free(nn->bias_scale);
    free(nn->neuron);
    free(nn->loss);
    free(nn->preact);
    free(nn->layer_type);
    free(nn->width);
    free(nn->activation);
    free(nn->config);
  }
  // Free the pooling argmax cache (independent of quantized state)
  if (nn->pool_argmax) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->pool_argmax[layer]);
    free(nn->pool_argmax);
  }
  free(nn);
}

nn_error_t nn_add_layer(nn_t *nn, layer_type_t layer_type, int width, int activation, void *config)
{
  cnn_t *cnn = NULL;
  pool_t *pool = NULL;
  int out_w = 0, out_h = 0; // CNN/POOL output spatial dims, computed once below

  // Validate the CNN config before mutating `nn` at all, so a rejected call
  // leaves the network completely unchanged (in particular, still safe to
  // keep using or to nn_free()).
  if (layer_type == LAYER_TYPE_CNN) {
    if (config == NULL) {
      return NN_ERROR_INVALID_CONFIG;
    }
    cnn_t *cnn_check = (cnn_t *)config;
    // padding and dilation are accepted and round-tripped through save/load,
    // but neither the output-size formula below nor nn_conv2d()'s actual
    // convolution loop implements them -- silently accepting a non-default
    // value would produce a different (and wrong, from the caller's
    // expectation) result instead of what was asked for. Reject rather than
    // silently ignore, until they're genuinely implemented.
    if (cnn_check->padding != 0 || cnn_check->dilation != 1) {
      fprintf(stderr, "nn_add_layer: CNN padding/dilation are not implemented (got padding=%u, dilation=%u; only padding=0, dilation=1 are supported)\n", cnn_check->padding, cnn_check->dilation);
      return NN_ERROR_INVALID_CONFIG;
    }
  }

  // Increase depth by one
  nn->depth++;
  nn->layer_type = (uint8_t *)realloc(nn->layer_type, nn->depth * sizeof(*nn->layer_type));
  if (nn->layer_type == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  // nn->depth - 1 is the index of the new layer that we are adding
  nn->layer_type[nn->depth - 1] = (uint8_t)layer_type;
  nn->width = (uint32_t *)realloc(nn->width, nn->depth * sizeof(*nn->width));
  if (nn->width == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->width[nn->depth - 1] = (uint32_t)width;
  if (layer_type == LAYER_TYPE_CNN) {
    // config's validity (non-NULL, padding=0, dilation=1) was already
    // checked above before any of the mutation up to this point.
    cnn = (cnn_t *)config;
    out_w = ((cnn->in_w - cnn->kernel_size) / cnn->stride) + 1;
    out_h = ((cnn->in_h - cnn->kernel_size) / cnn->stride) + 1;
    nn->width[nn->depth - 1] = cnn->out_channels * out_w * out_h;
  } else if (layer_type == LAYER_TYPE_POOL) {
    if (config == NULL) {
      return NN_ERROR_INVALID_CONFIG;
    }
    pool = (pool_t *)config;
    out_w = ((pool->in_w - pool->pool_size) / pool->stride) + 1;
    out_h = ((pool->in_h - pool->pool_size) / pool->stride) + 1;
    nn->width[nn->depth - 1] = pool->channels * out_w * out_h;
  }
  nn->activation = (uint8_t *)realloc(nn->activation, nn->depth * sizeof(*nn->activation));
  if (nn->activation == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->activation[nn->depth - 1] = (uint8_t)activation;
  nn->config = (void **)realloc(nn->config, nn->depth * sizeof(*nn->config));
  if (nn->config == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->config[nn->depth - 1] = NULL;
  if (layer_type == LAYER_TYPE_CNN) {
    nn->config[nn->depth - 1] = (void *)malloc(sizeof(cnn_t));
    if (nn->config[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    // Copy the CNN configuration
    memcpy(nn->config[nn->depth - 1], config, sizeof(cnn_t));
    // Cache the output dims so nn_conv2d() and nn_train() never need to
    // re-derive them via division on every sample/epoch.
    ((cnn_t *)nn->config[nn->depth - 1])->out_w = (uint16_t)out_w;
    ((cnn_t *)nn->config[nn->depth - 1])->out_h = (uint16_t)out_h;
  } else if (layer_type == LAYER_TYPE_POOL) {
    nn->config[nn->depth - 1] = (void *)malloc(sizeof(pool_t));
    if (nn->config[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    // Copy the pooling configuration
    memcpy(nn->config[nn->depth - 1], config, sizeof(pool_t));
    // Cache the output dims for the same reason as the CNN case above.
    ((pool_t *)nn->config[nn->depth - 1])->out_w = (uint16_t)out_w;
    ((pool_t *)nn->config[nn->depth - 1])->out_h = (uint16_t)out_h;
  }
  nn->neuron = (float **)realloc(nn->neuron, nn->depth * sizeof(float *));
  if (nn->neuron == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->loss = (float **)realloc(nn->loss, nn->depth * sizeof(float *));
  if (nn->loss == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->preact = (float **)realloc(nn->preact, nn->depth * sizeof(float *));
  if (nn->preact == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight = (float **)realloc(nn->weight, (nn->depth) * sizeof(float *));
  if (nn->weight == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight_adj = (float **)realloc(nn->weight_adj, (nn->depth) * sizeof(float *));
  if (nn->weight_adj == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight_scale = (float **)realloc(nn->weight_scale, (nn->depth) * sizeof(float *));
  if (nn->weight_scale == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->bias = (float **)realloc(nn->bias, (nn->depth) * sizeof(float *));
  if (nn->bias == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->bias_scale = (float *)realloc(nn->bias_scale, (nn->depth) * sizeof(float));
  if (nn->bias_scale == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->pool_argmax = (int **)realloc(nn->pool_argmax, (nn->depth) * sizeof(int *));
  if (nn->pool_argmax == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->pool_argmax[nn->depth - 1] = NULL;
  // For layer 0, we do not allocate neuron/loss/preact (input is provided externally)
  if (nn->depth > 1) {
    nn->neuron[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
    if (nn->neuron[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    nn->loss[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
    if (nn->loss[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    nn->preact[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
    if (nn->preact[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    if (layer_type == LAYER_TYPE_POOL) {
      // Pooling has no learnable parameters
      nn->weight[nn->depth - 1] = NULL;
      nn->weight_adj[nn->depth - 1] = NULL;
      nn->weight_scale[nn->depth - 1] = NULL;
      nn->bias[nn->depth - 1] = NULL;
      // MIN/MAX pooling need to remember which input position "won" each
      // output, so backprop can route the gradient to only that position.
      if (pool->pooling_type == POOLING_TYPE_MAX || pool->pooling_type == POOLING_TYPE_MIN) {
        nn->pool_argmax[nn->depth - 1] = (int *)malloc(nn->width[nn->depth - 1] * sizeof(int));
        if (nn->pool_argmax[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
      }
    } else {
      // CNN, FC, and OUTPUT layers all store weight/weight_adj as one flat,
      // row-major buffer of rows*row_len elements: a "row" is a kernel for
      // CNN layers (row_len = kernel_size^2, one bias per output channel)
      // or a neuron for FC/OUTPUT layers (row_len = previous layer's width,
      // one bias per neuron) -- see quantized_layer_shape().
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, nn->depth - 1, &rows, &row_len, &bias_count);
      float range;
      if (layer_type == LAYER_TYPE_CNN) {
        // Xavier (Glorot) initialisation: fan_in/fan_out count every
        // connection a kernel weight participates in, i.e. across all
        // input/output channels, not just its own row_len.
        range = sqrtf(6.0f / (float)(cnn->in_channels * row_len + cnn->out_channels * row_len));
      } else {
        // Xavier (Glorot) initialization
        range = sqrtf(6.0f / (float)(rows + row_len));
      }
      nn->weight[nn->depth - 1] = (float *)malloc((size_t)rows * row_len * sizeof(float));
      nn->weight_adj[nn->depth - 1] = (float *)malloc((size_t)rows * row_len * sizeof(float));
      nn->weight_scale[nn->depth - 1] = (float *)malloc((size_t)rows * sizeof(float));
      nn->bias[nn->depth - 1] = (float *)malloc((size_t)bias_count * sizeof(float));
      if (!nn->weight[nn->depth - 1] || !nn->weight_adj[nn->depth - 1] || !nn->weight_scale[nn->depth - 1] || !nn->bias[nn->depth - 1])
        return NN_ERROR_OUT_OF_MEMORY;
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < row_len; ++c) {
          nn->weight[nn->depth - 1][r * row_len + c] = range * 2.0f * ((rand() / (float)RAND_MAX) - 0.5f);
          nn->weight_adj[nn->depth - 1][r * row_len + c] = 0.0f;
        }
        nn->weight_scale[nn->depth - 1][r] = 0.0f; /* filled during quantise */
      }
      for (int b = 0; b < bias_count; ++b)
        nn->bias[nn->depth - 1][b] = 0.0f;
    }
  }
  return NN_ERROR_NONE;
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
  float err;

  if (nn->quantized) {
    // Cannot train a quantized network, so convert to a floating point model first.
    nn_dequantize(nn);
  }
  nn->neuron[0] = inputs;
  forward_propagation(nn);
  // Capture this sample's pre-update error now, while neuron[] still reflects
  // the forward pass above. This is the conventional "training loss" and lets
  // us avoid a second, redundant forward_propagation() call at the end of
  // this function (nn->neuron[] is not touched again until the next forward
  // pass, so this is equivalent to what a trailing nn_error() call would have
  // computed from the pre-update weights).
  i = (int)nn->depth - 1;
  err = 0.0f;
  for (j = 0; j < (int)nn->width[i]; j++) {
    err += error(targets[j], nn->neuron[i][j]);
  }
  // Perform back propagation using gradient descent, which is an optimization
  // algorithm that follows the negative gradient of the objective function to
  // find the minimum of the function. Start at the output layer, and work
  // backward toward the input layer, adjusting weights along the way. Calculate
  // the error aka loss aka delta at the output.
  // Compute output layer loss. This must include the output layer's own
  // activation derivative here (not later), so that loss[] always
  // uniformly represents -dE/d(preact) at every layer -- the propagation
  // step below then never needs to re-derive a layer's own derivative from
  // its neighbor's loss.
  for (j = 0; j < (int)nn->width[i]; j++) {
    nn->loss[i][j] = error_derivative(targets[j], nn->neuron[i][j]) * activation_function[nn->activation[i]](nn->preact[i][j], true);
  }
  // Backpropagate loss into earlier layers
  for (i = nn->depth - 2; i > 0; i--) {
    if (nn->layer_type[i + 1] == LAYER_TYPE_POOL) {
      // Pooling has no weight matrix -- route/distribute the gradient
      // directly according to the pooling type instead of the generic
      // weighted-sum formula below. Accumulate routed gradients straight
      // into loss[i] (zeroed first), then apply layer i's own activation
      // derivative in place, same as the generic case does.
      memset(nn->loss[i], 0, nn->width[i] * sizeof(float));
      nn_pool_backward(nn, i + 1, nn->loss[i]);
      for (j = 0; j < (int)nn->width[i]; j++) {
        nn->loss[i][j] *= activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    } else {
      // Layer i+1's flat weight buffer has a row per its own neuron, each
      // row_len = width[i] wide (its previous layer's width, i.e. ours).
      const int row_len = (int)nn->width[i];
      for (j = 0; j < (int)nn->width[i]; j++) {
        sum = 0.0f;
        for (k = 0; k < (int)nn->width[i + 1]; k++) {
          // loss[i+1][k] already has layer i+1's own activation derivative
          // baked in (it was applied when loss[i+1] was computed, whether
          // at output-layer init above or in this same branch one
          // recursion level up) -- do not re-apply it here.
          sum += nn->loss[i + 1][k] * nn->weight[i + 1][k * row_len + j];
        }
        // The chain rule dictates that we should multiply the summed loss by the
        // derivative of the activation at the current neuron, not only during
        // weight updates, but immediately when calculating loss[i][j].
        nn->loss[i][j] = sum * activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    }
  }
  // Update biases (gradient descent step)
  for (i = 1; i < (int)nn->depth; i++) {
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      cnn_t *cnn = nn->config[i];
      int out_c  = cnn->out_channels;
      int x_out  = cnn->out_w;
      int y_out  = cnn->out_h;
      int plane  = x_out * y_out;
      for (j = 0; j < out_c; ++j) {
        float db = 0.0f;
        for (k = 0; k < plane; ++k)
          db += nn->loss[i][j * plane + k];
        nn->bias[i][j] += db * rate;
      }
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL) {
      // Pooling has no bias
    } else {
        // FC / output layers
        for (j = 0; j < (int)nn->width[i]; j++)
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
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      // Convolution gradient
      cnn_t *cnn = nn->config[i];
      int in_c   = cnn->in_channels;
      int out_c  = cnn->out_channels;
      int ksize  = cnn->kernel_size;
      int x_out  = cnn->out_w;
      int y_out  = cnn->out_h;
      int plane_out = x_out * y_out;
      int in_plane  = cnn->in_w * cnn->in_h;
      for (int oc = 0; oc < out_c; ++oc) {
        for (int ic = 0; ic < in_c; ++ic) {
          float *adj = nn->weight_adj[i] + (oc * in_c + ic) * ksize * ksize;
          memset(adj, 0, ksize * ksize * sizeof(float));
          for (int oy = 0; oy < y_out; ++oy) {
            int in_y = oy * cnn->stride;
            for (int ox = 0; ox < x_out; ++ox) {
              int in_x = ox * cnn->stride;
              float delta = nn->loss[i][oc * plane_out + oy * x_out + ox];
              const float *src = nn->neuron[i - 1] + ic * in_plane + in_y * cnn->in_w + in_x;
              for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                  adj[ky * ksize + kx] += delta * src[ky * cnn->in_w + kx];
                }
              }
            }
          }
        }
      }
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL) {
      // Pooling has no weights
    } else {
      // FC / output layers
      const int row_len = (int)nn->width[i - 1];
      for (j = 0; j < (int)nn->width[i]; j++)
        for (k = 0; k < row_len; k++)
          nn->weight_adj[i][j * row_len + k] = nn->loss[i][j] * nn->neuron[i - 1][k];
    }
  }
  // Apply weight adjustments
  for (i = (int)nn->depth - 1; i > 0; i--) {
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      cnn_t *cnn = nn->config[i];
      int kernels = cnn->out_channels * cnn->in_channels;
      int k_elems = cnn->kernel_size * cnn->kernel_size;
      int total = kernels * k_elems;
      for (int idx = 0; idx < total; ++idx)
        nn->weight[i][idx] += nn->weight_adj[i][idx] * rate;
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL) {
      // Pooling has no weights
    } else {
      // FC / output layers
      int total = (int)nn->width[i] * (int)nn->width[i - 1];
      for (int idx = 0; idx < total; ++idx)
        nn->weight[i][idx] += nn->weight_adj[i][idx] * rate;
    }
  }
  // Return the pre-update error computed above
  return err;
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
    // peek layer_type first to know if cnn_t follows
    if (fscanf(file, "%d", &layer_type) != 1) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
    if (fscanf(file, "%d %d", &w, &act) != 2) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
    cnn_t ctmp;
    pool_t ptmp;
    void *cptr = NULL;
    if (layer_type == LAYER_TYPE_CNN) {
      if (fscanf(file, " %hu %hu %hhu %hhu %hhu %hhu %hhu %hhu", &ctmp.in_h, &ctmp.in_w, &ctmp.in_channels, &ctmp.out_channels, &ctmp.kernel_size, &ctmp.stride, &ctmp.padding, &ctmp.dilation) != 8) {
        fclose(file);
        nn_free(nn);
        return NULL;
      }
      cptr = &ctmp;
      // nn_add_layer will recompute width
      w = 0;
    } else if (layer_type == LAYER_TYPE_POOL) {
      int pt;
      if (fscanf(file, " %hu %hu %hhu %hhu %hhu %d", &ptmp.in_h, &ptmp.in_w, &ptmp.channels, &ptmp.pool_size, &ptmp.stride, &pt) != 6) {
        fclose(file);
        nn_free(nn);
        return NULL;
      }
      ptmp.pooling_type = (pooling_type_t)pt;
      cptr = &ptmp;
      // nn_add_layer will recompute width
      w = 0;
    }
    // Consume '\n'
    fgetc(file);
    if (nn_add_layer(nn, layer_type, w, act, cptr) != 0) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
  }
  // Note: neuron/loss/preact for every layer >= 1 were already allocated by
  // nn_add_layer() inside the layer-construction loop above; re-allocating
  // them here would just leak those allocations.
  // If float mode, read floats into nn->weight / nn->bias
  if (!nn->quantized) {
    for (int layer = 1; layer < nn->depth; layer++) {
      float dummy_scale;
      // Skip bias_scale (0)
      if (fscanf(file, "%f\n", &dummy_scale) != 1) {
        goto cleanup_float;
      }
      if (nn->layer_type[layer] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[layer];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        // Extra dummy line that the saver emits before the kernels
        if (fscanf(file, "%f\n", &dummy_scale) != 1)
          goto cleanup_float;
        // kernels
        for (int k = 0; k < kernels; ++k) {
          // per-kernel weight-scale placeholder
          if (fscanf(file, "%f\n", &dummy_scale) != 1)
            goto cleanup_float;
          for (int e = 0; e < k_elems; ++e) {
            if (fscanf(file, "%f\n", &nn->weight[layer][k * k_elems + e]) != 1)
              goto cleanup_float;
          }
        }
        // One bias per output channel
        for (int oc = 0; oc < c->out_channels; ++oc) {
          if (fscanf(file, "%f\n", &nn->bias[layer][oc]) != 1)
            goto cleanup_float;
        }
      } else if (nn->layer_type[layer] == LAYER_TYPE_POOL) {
        // Pooling has no weights/bias beyond the placeholder line already consumed above
      } else {
        // Fully-connected / output layer
        int row_len = (int)nn->width[layer - 1];
        for (int i = 0; i < (int)nn->width[layer]; i++) {
          // Skip weight_scale (0)
          if (fscanf(file, "%f\n", &dummy_scale) != 1)
            goto cleanup_float;
          for (int j = 0; j < row_len; j++) {
            if (fscanf(file, "%f\n", &nn->weight[layer][i * row_len + j]) != 1)
              goto cleanup_float;
          }
          if (fscanf(file, "%f\n", &nn->bias[layer][i]) != 1)
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
  // Free all float-side allocations made by nn_add_layer (weight, weight_adj, bias)
  for (int layer = 1; layer < nn->depth; layer++) {
    // weight/weight_adj are each one flat buffer per layer (or NULL for
    // POOL); free(NULL) is a no-op.
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
  for (int layer = 1; layer < nn->depth; layer++)
    free(nn->weight_scale[layer]);
  free(nn->weight_scale);
  free(nn->bias_scale);
  nn->weight_scale = NULL;
  nn->bias_scale = NULL;
  // Allocate top-level arrays for quantized model
  nn->weight_quantized = (int8_t **)malloc(sizeof(int8_t *) * nn->depth);
  nn->weight_scale = (float **)malloc(sizeof(float *) * nn->depth);
  nn->bias_quantized = (int8_t **)malloc(sizeof(int8_t *) * nn->depth);
  nn->bias_scale = (float *)malloc(sizeof(float) * nn->depth);
  if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized || !nn->bias_scale) {
    goto cleanup_quant_top;
  }
  // Initialize layer 0 entries
  nn->weight_quantized[0] = NULL;
  nn->weight_scale[0] = NULL;
  nn->bias_quantized[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  // Read quantized data, layer by layer. Each layer's weight_quantized is
  // now one flat, row-major buffer (not an array of per-row pointers), so
  // there is no per-row allocation bookkeeping to unwind on failure --
  // every layer's fields are either NULL or one single valid allocation.
  int layer = 0;
  for (layer = 1; layer < nn->depth; layer++) {
    if (nn->layer_type[layer] == LAYER_TYPE_POOL) {
      // Pooling has no weights/bias to read
      nn->weight_quantized[layer] = NULL;
      nn->weight_scale[layer] = NULL;
      nn->bias_quantized[layer] = NULL;
      nn->bias_scale[layer] = 0.0f;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, layer, &rows, &row_len, &bias_count);
    nn->weight_quantized[layer] = (int8_t *)malloc((size_t)rows * row_len * sizeof(int8_t));
    nn->weight_scale[layer] = (float *)malloc(sizeof(float) * rows);
    nn->bias_quantized[layer] = (int8_t *)malloc(sizeof(int8_t) * bias_count);
    if (!nn->weight_quantized[layer] || !nn->weight_scale[layer] || !nn->bias_quantized[layer]) {
      goto cleanup_quant_per_layer;
    }
    // For each row (neuron or kernel), read weight_scale + quantized weights
    for (int row = 0; row < rows; row++) {
      if (fscanf(file, "%f\n", &nn->weight_scale[layer][row]) != 1) {
        goto cleanup_quant_per_layer;
      }
      for (int w = 0; w < row_len; w++) {
        int int_w;
        if (fscanf(file, "%d\n", &int_w) != 1) {
          goto cleanup_quant_per_layer;
        }
        nn->weight_quantized[layer][row * row_len + w] = (int8_t)int_w;
      }
    }
    // Read bias_scale[layer]
    if (fscanf(file, "%f\n", &nn->bias_scale[layer]) != 1) {
      goto cleanup_quant_per_layer;
    }
    // Read each bias (one per output channel for CNN, one per neuron for FC/OUTPUT)
    for (int b = 0; b < bias_count; b++) {
      int int_b;
      if (fscanf(file, "%d\n", &int_b) != 1) {
        goto cleanup_quant_per_layer;
      }
      nn->bias_quantized[layer][b] = (int8_t)int_b;
    }
  }
  fclose(file);
  return nn;
  // If we failed before allocating the top-level quantized arrays:
cleanup_quant_top:
  free(nn->weight_quantized);
  free(nn->weight_scale);
  free(nn->bias_quantized);
  free(nn->bias_scale);
  // NULL these out before nn_free(): nn_free()'s quantized branch would
  // otherwise index through these same (now-freed) top-level pointers
  // again, since it doesn't know we already tore them down here.
  nn->weight_quantized = NULL;
  nn->weight_scale = NULL;
  nn->bias_quantized = NULL;
  nn->bias_scale = NULL;
  fclose(file);
  nn_free(nn);
  return NULL;
  // If we failed partway through allocating/reading layer `layer` (or an
  // earlier layer had already been fully read): every layer 1..layer is
  // either NULL or a single valid flat allocation, so a plain free() over
  // each (safe as a no-op on NULL) is enough -- no per-row unwinding needed.
cleanup_quant_per_layer:
  for (int L = 1; L <= layer && L < nn->depth; L++) {
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
  // Magic number
  uint8_t magic[NN_BINARY_MAGIC_LEN];
  if (fread(magic, 1, NN_BINARY_MAGIC_LEN, file) != NN_BINARY_MAGIC_LEN ||
      memcmp(magic, NN_BINARY_MAGIC, NN_BINARY_MAGIC_LEN) != 0) {
    fclose(file);
    return NULL;
  }
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
    uint8_t layer_type;
    uint32_t w;
    uint8_t a;
    if (fread(&layer_type, sizeof(layer_type), 1, file) != 1)
      goto error;
    if (fread(&w, sizeof(w), 1, file) != 1)
      goto error;
    if (fread(&a, sizeof(a), 1, file) != 1)
      goto error;
    cnn_t ctmp;
    pool_t ptmp;
    void *cptr = NULL;
    if (layer_type == LAYER_TYPE_CNN) {
      if (fread(&ctmp, sizeof(ctmp), 1, file) != 1)
        goto error;
      cptr = &ctmp; w = 0;
    } else if (layer_type == LAYER_TYPE_POOL) {
      if (fread(&ptmp, sizeof(ptmp), 1, file) != 1)
        goto error;
      cptr = &ptmp; w = 0;
    }
    if (nn_add_layer(nn, layer_type, (int)w, (int)a, cptr) != 0)
     goto cleanup;
  }
  // Note: neuron/loss/preact for every layer >= 1 were already allocated by
  // nn_add_layer() inside the layer-construction loop above; re-allocating
  // them here would just leak those allocations.
  // Read weights & biases
  if (!nn->quantized) {
    // Float-mode: read dummy scales + real floats
    for (int L = 1; L < (int)depth; L++) {
      float dummy;
      // bias_scale placeholder
      if (fread(&dummy, sizeof(dummy), 1, file) != 1)
        goto cleanup;
      if (nn->layer_type[L] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[L];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        for (int k = 0; k < kernels; ++k) {
          // per-kernel weight-scale placeholder
          if (fread(&dummy, sizeof(dummy), 1, file) != 1)
            goto cleanup;
          if (fread(nn->weight[L] + k * k_elems, sizeof(float), k_elems, file) != (size_t)k_elems)
            goto cleanup;
        }
        // One bias per output channel
        if (fread(nn->bias[L], sizeof(float), c->out_channels, file) != (size_t)c->out_channels)
          goto cleanup;
      } else if (nn->layer_type[L] == LAYER_TYPE_POOL) {
        // Pooling has no weights/bias beyond the placeholder read above
      } else {
        uint32_t curr = nn->width[L], prev = nn->width[L - 1];
        for (uint32_t i = 0; i < curr; i++) {
          // weight_scale placeholder
          if (fread(&dummy, sizeof(dummy), 1, file) != 1)
            goto cleanup;
          // weights
          if (fread(nn->weight[L] + i * prev, sizeof(float), prev, file) != prev)
            goto cleanup;
          // bias
          if (fread(&nn->bias[L][i], sizeof(float), 1, file) != 1)
            goto cleanup;
        }
      }
    }
  } else {
    // Quantized-mode: free float-side, allocate quantized arrays
    for (int L = 1; L < (int)depth; L++) {
      // weight/weight_adj are each one flat buffer per layer (or NULL for
      // POOL); free(NULL) is a no-op.
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
    for (int L = 1; L < (int)depth; L++)
      free(nn->weight_scale[L]);
    free(nn->weight_scale);
    nn->weight_scale = NULL;
    free(nn->bias_scale);
    nn->bias_scale = NULL;
    // Top-level quant arrays
    nn->weight_quantized = malloc(depth * sizeof(int8_t *));
    nn->weight_scale = malloc(depth * sizeof(float *));
    nn->bias_quantized = malloc(depth * sizeof(int8_t *));
    nn->bias_scale = malloc(depth * sizeof(float));
    if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized ||
        !nn->bias_scale)
      goto cleanup;
    // Zero every layer's slot up front so that if a `goto cleanup` below
    // fires partway through the per-layer read loop, nn_free() can safely
    // free every layer -- including ones not reached yet -- instead of
    // indexing uninitialized garbage left over from this malloc.
    memset(nn->weight_quantized, 0, depth * sizeof(int8_t *));
    memset(nn->weight_scale, 0, depth * sizeof(float *));
    memset(nn->bias_quantized, 0, depth * sizeof(int8_t *));
    nn->bias_scale[0] = 0.0f;
    // Read per-layer quant data
    for (int L = 1; L < (int)depth; L++) {
      if (nn->layer_type[L] == LAYER_TYPE_POOL) {
        // Pooling has no weights/bias to read (already NULL from the memsets above)
        nn->bias_scale[L] = 0.0f;
        continue;
      }
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
      // Allocate one flat, row-major buffer per layer
      nn->weight_quantized[L] = malloc((size_t)rows * row_len * sizeof(int8_t));
      nn->weight_scale[L] = malloc(rows * sizeof(float));
      nn->bias_quantized[L] = malloc(bias_count * sizeof(int8_t));
      if (!nn->weight_quantized[L] || !nn->weight_scale[L] ||
          !nn->bias_quantized[L])
        goto cleanup;
      // Read each row's (neuron or kernel) weight_scale and weights
      for (int i = 0; i < rows; i++) {
        if (fread(&nn->weight_scale[L][i], sizeof(float), 1, file) != 1)
          goto cleanup;
        if (fread(nn->weight_quantized[L] + i * row_len, sizeof(int8_t), row_len, file) !=
            (size_t)row_len)
          goto cleanup;
      }
      // Read bias_scale[L]
      if (fread(&nn->bias_scale[L], sizeof(float), 1, file) != 1)
        goto cleanup;
      // Read quantized biases (one per output channel for CNN, one per neuron for FC/OUTPUT)
      if (fread(nn->bias_quantized[L], sizeof(int8_t), bias_count, file) != (size_t)bias_count)
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
nn_error_t nn_save_model_ascii(nn_t *nn, const char *path)
{
  FILE *file = fopen(path, "w");
  if (file == NULL)
    return NN_ERROR_FILE_WRITE;
  // Write quantized flag: 0 = floating, 1 = fixed point
  fprintf(file, "%d\n", nn->quantized ? 1 : 0);
  // Write model version (major, minor, patch, build)
  fprintf(file, "%hhu %hhu %hhu %hhu\n", nn->version_major, nn->version_minor, nn->version_patch, nn->version_build);
  // Write depth
  fprintf(file, "%" PRId32 "\n", nn->depth);
  // Write each layer's type, width, and activation
  for (int i = 0; i < (int)nn->depth; i++) {
    // fprintf(file, "%d %" PRId32 " %d\n", nn->layer_type[i], nn->width[i], nn->activation[i]);
    fprintf(file, "%d %" PRId32 " %d", nn->layer_type[i], nn->width[i], nn->activation[i]);
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      cnn_t *c = nn->config[i];
      fprintf(file, " %d %d %d %d %d %d %d %d", c->in_h, c->in_w, c->in_channels, c->out_channels, c->kernel_size, c->stride, c->padding, c->dilation);
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL) {
      pool_t *p = nn->config[i];
      fprintf(file, " %d %d %d %d %d %d", p->in_h, p->in_w, p->channels, p->pool_size, p->stride, (int)p->pooling_type);
    }
    fputc('\n', file);
  }
  // Write weights & biases
  if (!nn->quantized) {
    // Float mode: write weight_scale (0), weights, and bias
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      // bias_scale placeholder
      fprintf(file, "0\n");
      if (nn->layer_type[layer] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[layer];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        // One scale + k² weights per kernel
        // Bias scale placeholder
        fprintf(file, "0\n");
        for (int k = 0; k < kernels; ++k) {
          // Weight scale
          fprintf(file, "0\n");
          for (int e = 0; e < k_elems; ++e)
            fprintf(file, "%f\n", nn->weight[layer][k * k_elems + e]);
        }
        for (int oc = 0; oc < c->out_channels; ++oc)
          fprintf(file, "%f\n", nn->bias[layer][oc]);
      } else if (nn->layer_type[layer] == LAYER_TYPE_POOL) {
        // Pooling has no weights/bias beyond the placeholder line already written above
      } else {
        // FC / output
        int row_len = (int)nn->width[layer - 1];
        for (int i = 0; i < (int)nn->width[layer]; i++) {
          // weight_scale placeholder
          fprintf(file, "0\n");
          for (int j = 0; j < row_len; j++) {
            fprintf(file, "%f\n", nn->weight[layer][i * row_len + j]);
          }
          fprintf(file, "%f\n", nn->bias[layer][i]);
        }
      }
    }
  } else {
    // Quantized mode: write weight_scale, quantized weights, bias_scale, quantized bias
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      if (nn->layer_type[layer] == LAYER_TYPE_POOL) {
        // Pooling has no weights/bias to write
        continue;
      }
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, layer, &rows, &row_len, &bias_count);
      for (int row = 0; row < rows; row++) {
        fprintf(file, "%f\n", nn->weight_scale[layer][row]);
        for (int w = 0; w < row_len; w++) {
          fprintf(file, "%d\n", (int)nn->weight_quantized[layer][row * row_len + w]);
        }
      }
      fprintf(file, "%f\n", nn->bias_scale[layer]);
      for (int b = 0; b < bias_count; b++) {
        fprintf(file, "%d\n", (int)nn->bias_quantized[layer][b]);
      }
    }
  }
  fclose(file);
  return NN_ERROR_NONE;
}

// Exports a neural net model as raw binary
nn_error_t nn_save_model_binary(nn_t *nn, const char *path)
{
  FILE *file = fopen(path, "wb");
  if (!file)
    return NN_ERROR_FILE_WRITE;
  // Magic number
  fwrite(NN_BINARY_MAGIC, 1, NN_BINARY_MAGIC_LEN, file);
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
    if (layer_type == LAYER_TYPE_CNN) {
      cnn_t *c = nn->config[i];
      // Struct is POD -> dump
      fwrite(c, sizeof(cnn_t), 1, file);
    } else if (layer_type == LAYER_TYPE_POOL) {
      pool_t *p = nn->config[i];
      fwrite(p, sizeof(pool_t), 1, file);
    }
  }
  // Weights & biases
  if (!nn->quantized) {
    // Float mode: placeholders for scales and actual floats
    for (uint32_t L = 1; L < depth; L++) {
      float bias_scale = 0.0f;
      fwrite(&bias_scale, sizeof(bias_scale), 1, file);
      if (nn->layer_type[L] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[L];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        for (int k = 0; k < kernels; ++k) {
          float weight_scale = 0.0f;
          fwrite(&weight_scale, sizeof(weight_scale), 1, file);
          fwrite(nn->weight[L] + k * k_elems, sizeof(float), k_elems, file);
        }
        fwrite(nn->bias[L], sizeof(float), c->out_channels, file);
      } else if (nn->layer_type[L] == LAYER_TYPE_POOL) {
        // Pooling has no weights/bias beyond the placeholder written above
      } else {
        uint32_t curr = nn->width[L], prev = nn->width[L - 1];
        for (uint32_t i = 0; i < curr; i++) {
          float weight_scale = 0.0f;
          fwrite(&weight_scale, sizeof(weight_scale), 1, file);
          fwrite(nn->weight[L] + i * prev, sizeof(float), prev, file);
          fwrite(&nn->bias[L][i], sizeof(float), 1, file);
        }
      }
    }
  } else {
    // Quantized mode: real scales and int8 quantized data
    for (uint32_t L = 1; L < depth; L++) {
      if (nn->layer_type[L] == LAYER_TYPE_POOL) {
        // Pooling has no weights/bias to write
        continue;
      }
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
      for (int i = 0; i < rows; i++) {
        // Per-row (neuron or kernel) weight scale
        fwrite(&nn->weight_scale[L][i], sizeof(float), 1, file);
        // Quantized weights
        fwrite(nn->weight_quantized[L] + i * row_len, sizeof(int8_t), row_len, file);
      }
      // Bias scale (one per layer)
      fwrite(&nn->bias_scale[L], sizeof(float), 1, file);
      // Quantized biases (one per output channel for CNN, one per neuron for FC/OUTPUT)
      fwrite(nn->bias_quantized[L], sizeof(int8_t), bias_count, file);
    }
  }
  fclose(file);
  return NN_ERROR_NONE;
}

// Loads a model file, auto-detecting ascii vs. binary by peeking for the
// binary format's magic number.
nn_t *nn_load_model(const char *path)
{
  FILE *file = fopen(path, "rb");
  if (!file)
    return NULL;
  uint8_t magic[NN_BINARY_MAGIC_LEN];
  bool is_binary = fread(magic, 1, NN_BINARY_MAGIC_LEN, file) == NN_BINARY_MAGIC_LEN &&
                    memcmp(magic, NN_BINARY_MAGIC, NN_BINARY_MAGIC_LEN) == 0;
  fclose(file);
  return is_binary ? nn_load_model_binary(path) : nn_load_model_ascii(path);
}

// Saves a model file, writing binary format if `path` ends in ".bin"
// (case-insensitive) and ascii format otherwise.
nn_error_t nn_save_model(nn_t *nn, const char *path)
{
  size_t path_len = strlen(path);
  static const char ext[] = ".bin";
  size_t ext_len = sizeof(ext) - 1;
  bool is_binary = path_len >= ext_len;
  for (size_t i = 0; is_binary && i < ext_len; i++)
    is_binary = tolower((unsigned char)path[path_len - ext_len + i]) == ext[i];
  return is_binary ? nn_save_model_binary(nn, path) : nn_save_model_ascii(nn, path);
}

nn_error_t nn_remove_neuron(nn_t *nn, int layer, int neuron_index)
{
  if (nn == NULL || layer <= 0 || layer >= (int)nn->depth || neuron_index < 0 || neuron_index >= (int)nn->width[layer]) {
    return NN_ERROR_INVALID_ARGUMENT;
  }
  // A CNN/POOL layer's width is derived entirely from its cnn_t/pool_t
  // config (spatial dims x channels), not a flat list of independent
  // neurons, and its weight array (if any) isn't neuron-indexed -- there is
  // no well-defined way to "remove one neuron" from it without corrupting
  // the layer's structural computation. Likewise, if the NEXT layer is
  // CNN/POOL, its weight array has no per-input-neuron column to shrink.
  if (nn->layer_type[layer] == LAYER_TYPE_CNN || nn->layer_type[layer] == LAYER_TYPE_POOL) {
    return NN_ERROR_UNSUPPORTED_LAYER;
  }
  if (layer + 1 < (int)nn->depth &&
      (nn->layer_type[layer + 1] == LAYER_TYPE_CNN || nn->layer_type[layer + 1] == LAYER_TYPE_POOL)) {
    return NN_ERROR_UNSUPPORTED_LAYER;
  }
  int old_width = nn->width[layer];
  // Both `layer` and (if present) `layer + 1` are guaranteed FC/OUTPUT by
  // the guards above, so their weight buffers are always the generic flat,
  // row-major [width[L] x width[L-1]] layout -- no CNN/POOL shape to worry
  // about here.
  int in_row_len = (int)nn->width[layer - 1]; // this layer's row length (unaffected by removing a row)
  // Shift out neuron / preact / loss in this layer
  memmove(&nn->neuron[layer][neuron_index], &nn->neuron[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  memmove(&nn->preact[layer][neuron_index], &nn->preact[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  memmove(&nn->loss[layer][neuron_index], &nn->loss[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  // Remove row `neuron_index` (in_row_len elements) from this layer's own
  // flat weight buffer, then shrink the buffer to match.
  if (nn->quantized) {
    int8_t *wq = nn->weight_quantized[layer];
    memmove(wq + neuron_index * in_row_len, wq + (neuron_index + 1) * in_row_len,
            sizeof(int8_t) * (size_t)(old_width - neuron_index - 1) * in_row_len);
    nn->weight_quantized[layer] = (int8_t *)realloc(wq, sizeof(int8_t) * (size_t)(old_width - 1) * in_row_len);
    // Shift the single byte biases in bias_quantized[layer]
    memmove(&nn->bias_quantized[layer][neuron_index], &nn->bias_quantized[layer][neuron_index + 1], sizeof(int8_t) * (old_width - neuron_index - 1));
    nn->bias_quantized[layer] = (int8_t *)realloc(nn->bias_quantized[layer], sizeof(int8_t) * (old_width - 1));
    // Leave bias_scale[layer] alone (it's a single float per layer).
    // The float side arrays (weight[layer], weight_adj[layer], bias[layer]) are NULL here, so we must NOT touch them in quantized mode.
  } else {
    float *w = nn->weight[layer];
    memmove(w + neuron_index * in_row_len, w + (neuron_index + 1) * in_row_len,
            sizeof(float) * (size_t)(old_width - neuron_index - 1) * in_row_len);
    nn->weight[layer] = (float *)realloc(w, sizeof(float) * (size_t)(old_width - 1) * in_row_len);
    // weight_adj's contents are always fully overwritten before being read
    // again (see nn_train()'s weight-adjustment computation, which covers
    // every element every call), so it only needs to end up the right
    // *size* -- no need to shift its (disposable) contents first.
    nn->weight_adj[layer] = (float *)realloc(nn->weight_adj[layer], sizeof(float) * (size_t)(old_width - 1) * in_row_len);
    // Shift the float biases in bias[layer]
    memmove(&nn->bias[layer][neuron_index], &nn->bias[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
    nn->bias[layer] = (float *)realloc(nn->bias[layer], sizeof(float) * (old_width - 1));
  }
  // Update next layer's weights to remove the input connection from this
  // neuron: every row of layer+1's weight matrix must have column
  // `neuron_index` removed, shrinking its stride from old_width to
  // old_width-1. Since the buffer is row-major and the new stride is
  // smaller, this is done in place, row by row in increasing order: first
  // splice the column out of the row (in place, at its old offset), then
  // slide the now-shorter row down to its new tighter offset. Processing in
  // increasing order guarantees the slide-down for earlier rows never
  // reaches into a later row's not-yet-read data (row r's old data starts
  // at r*old_width, while all earlier rows' slides finish by r*(old_width-1),
  // which is always <= r*old_width).
  if (layer + 1 < (int)nn->depth) {
    int old_prev_width = old_width;
    int new_prev_width = old_width - 1;
    int next_width = nn->width[layer + 1];
    if (nn->quantized) {
      int8_t *wq_next = nn->weight_quantized[layer + 1];
      for (int r = 0; r < next_width; r++) {
        int8_t *old_row = wq_next + (size_t)r * old_prev_width;
        memmove(old_row + neuron_index, old_row + neuron_index + 1, sizeof(int8_t) * (old_prev_width - neuron_index - 1));
        memmove(wq_next + (size_t)r * new_prev_width, old_row, sizeof(int8_t) * new_prev_width);
      }
      nn->weight_quantized[layer + 1] = (int8_t *)realloc(wq_next, sizeof(int8_t) * (size_t)next_width * new_prev_width);
      // Do NOT touch any biases in layer+1.
    } else {
      float *w_next = nn->weight[layer + 1];
      for (int r = 0; r < next_width; r++) {
        float *old_row = w_next + (size_t)r * old_prev_width;
        memmove(old_row + neuron_index, old_row + neuron_index + 1, sizeof(float) * (old_prev_width - neuron_index - 1));
        memmove(w_next + (size_t)r * new_prev_width, old_row, sizeof(float) * new_prev_width);
      }
      nn->weight[layer + 1] = (float *)realloc(w_next, sizeof(float) * (size_t)next_width * new_prev_width);
      // weight_adj's contents are disposable (see above) -- just resize it.
      nn->weight_adj[layer + 1] = (float *)realloc(nn->weight_adj[layer + 1], sizeof(float) * (size_t)next_width * new_prev_width);
      // Do NOT touch any biases in layer+1.
    }
  }
  // Decrement the width of this layer
  nn->width[layer] = old_width - 1;
  return NN_ERROR_NONE;
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
  // CNN/POOL layers aren't neuron-indexed the way this function assumes
  // (see nn_remove_neuron() for why); there's no meaningful "neuron weight"
  // to report for one.
  if (nn->layer_type[layer] == LAYER_TYPE_CNN || nn->layer_type[layer] == LAYER_TYPE_POOL) {
    return 0.0f;
  }
  float total = 0.0f;
  // Sum absolute values of input weights (previous layer to this neuron)
  int in_row_len = (int)nn->width[layer - 1];
  for (int i = 0; i < in_row_len; i++) {
    if (nn->quantized) {
      // For quantized models, we use the quantized weights
      total += fabsf((float)nn->weight_quantized[layer][neuron_index * in_row_len + i] * nn->weight_scale[layer][neuron_index]);
    } else {
      // For float models, we use the float weights directly
      total += fabsf(nn->weight[layer][neuron_index * in_row_len + i]);
    }
  }
  // Sum absolute values of output weights (this neuron to next layer), only
  // when the next layer's weight array is itself neuron-indexed (FC/OUTPUT).
  if (layer + 1 < (int)nn->depth &&
      nn->layer_type[layer + 1] != LAYER_TYPE_CNN && nn->layer_type[layer + 1] != LAYER_TYPE_POOL) {
    int next_row_len = (int)nn->width[layer]; // layer+1's row length == this layer's width
    for (int i = 0; i < (int)nn->width[layer + 1]; i++) {
      if (nn->quantized) {
        // For quantized models, we use the quantized weights
        total += fabsf((float)nn->weight_quantized[layer + 1][i * next_row_len + neuron_index] * nn->weight_scale[layer + 1][i]);
      } else {
        // For float models, we use the float weights directly
        total += fabsf(nn->weight[layer + 1][i * next_row_len + neuron_index]);
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
  // Search all hidden layers (1..depth-2), skipping CNN/POOL layers -- their
  // width isn't a flat list of independent neurons, so they can't be pruned
  // this way (see nn_remove_neuron()).
  for (int layer = 1; layer < (int)nn->depth - 1; layer++) {
    if (nn->layer_type[layer] == LAYER_TYPE_CNN || nn->layer_type[layer] == LAYER_TYPE_POOL) {
      continue;
    }
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

void nn_pool2d(char *src, char *dest, int filter_size, int stride, pooling_type_t pooling_type, int x_in, int y_in)
{
  uint32_t pool_value;
  uint32_t pool_value_temp;

  int x_out = ((x_in - filter_size) / stride) + 1;
  int y_out = ((y_in - filter_size) / stride) + 1;
  // Assume src and dest are RGBA (4 channels)
  for (int z = 0; z < 4; z++) {
    for (int y = 0; y < y_out; y++) {
      for (int x = 0; x < x_out; x++) {
        switch (pooling_type) {
        case POOLING_TYPE_MIN:
          pool_value = 255;
          for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
              pool_value_temp = (uint8_t)*(src + z + ((x + fx) * stride) * 4 + ((y + fy) * stride) * 4 * x_in);
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
              pool_value_temp = (uint8_t)*(src + z + ((x + fx) * stride) * 4 + ((y + fy) * stride) * 4 * x_in);
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
              pool_value += (uint8_t)*(src + z + ((x + fx) * stride) * 4 + ((y + fy) * stride) * 4 * x_in);
            }
          }
          pool_value /= (filter_size * filter_size);
          break;
        case POOLING_TYPE_NONE:
        default:
          pool_value = 0;
          break;
        }
        *(dest + z + (x * 4) + (y * x_out * 4)) = (char)pool_value;
      }
    }
  }
}

// 2D convolution
void nn_conv2d(nn_t *nn, int layer)
{
    // Derive channel counts from previous bookkeeping
    cnn_t *cnn = nn->config[layer];
    const int in_c = cnn->in_channels;
    const int x_out = cnn->out_w;
    const int y_out = cnn->out_h;
    const int plane_out = y_out * x_out;
    const int out_c = cnn->out_channels;
    const int row_len = cnn->kernel_size * cnn->kernel_size; // flat weight buffer stride per kernel
    // Sanity‑check shapes
    if ((in_c <= 0) || (out_c <= 0) ||
        ((uint32_t)out_c * plane_out != nn->width[layer])) {
        fprintf(stderr, "conv2d: inconsistent shape (in_c=%d, out_c=%d, plane_out=%d, width[%d]=%u)\n", in_c, out_c, plane_out, layer, nn->width[layer]);
        return;
    }
    // Perform the convolution. The quantized/float branch is hoisted out to
    // once per layer call (instead of once per output-channel/pixel/input-
    // channel), and in the quantized case the per-input-channel weight_scale
    // multiply happens once after accumulating that channel's raw kernel_size^2
    // int8*float products, instead of once per kernel tap.
    if (nn->quantized) {
        for (int oc = 0; oc < out_c; ++oc) {
            const float bias = (float)nn->bias_quantized[layer][oc] * nn->bias_scale[layer];
            float *dst = nn->neuron[layer] + oc * plane_out;
            float *pre = nn->preact[layer] + oc * plane_out;
            for (int oy = 0; oy < y_out; ++oy) {
                const int in_y = oy * cnn->stride;
                for (int ox = 0; ox < x_out; ++ox) {
                    const int in_x = ox * cnn->stride;
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_c; ++ic) {
                        const float *src = nn->neuron[layer - 1] + ic * cnn->in_h * cnn->in_w + in_y * cnn->in_w + in_x;
                        const int8_t *kptr = nn->weight_quantized[layer] + (oc * in_c + ic) * row_len;
                        const float wsc = nn->weight_scale[layer][oc * in_c + ic];
                        const float *sptr = src;
                        float raw = 0.0f;
                        for (int ky = 0; ky < cnn->kernel_size; ++ky) {
                            for (int kx = 0; kx < cnn->kernel_size; ++kx)
                                raw += sptr[kx] * (float)kptr[kx];
                            sptr += cnn->in_w;
                            kptr += cnn->kernel_size;
                        }
                        sum += raw * wsc;
                    }
                    const int oidx = oy * x_out + ox;
                    sum += bias;
                    pre[oidx] = sum;
                    dst[oidx] = activation_function[nn->activation[layer]](sum, false);
                }
            }
        }
    } else {
        for (int oc = 0; oc < out_c; ++oc) {
            const float bias = nn->bias[layer][oc];
            float *dst = nn->neuron[layer] + oc * plane_out;
            float *pre = nn->preact[layer] + oc * plane_out;
            for (int oy = 0; oy < y_out; ++oy) {
                const int in_y = oy * cnn->stride;
                for (int ox = 0; ox < x_out; ++ox) {
                    const int in_x = ox * cnn->stride;
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_c; ++ic) {
                        const float *src = nn->neuron[layer - 1] + ic * cnn->in_h * cnn->in_w + in_y * cnn->in_w + in_x;
                        const float *kptr = nn->weight[layer] + (oc * in_c + ic) * row_len;
                        const float *sptr = src;
                        for (int ky = 0; ky < cnn->kernel_size; ++ky) {
                            for (int kx = 0; kx < cnn->kernel_size; ++kx)
                                sum += sptr[kx] * kptr[kx];
                            sptr += cnn->in_w;
                            kptr += cnn->kernel_size;
                        }
                    }
                    const int oidx = oy * x_out + ox;
                    sum += bias;
                    pre[oidx] = sum;
                    dst[oidx] = activation_function[nn->activation[layer]](sum, false);
                }
            }
        }
    }
}

// In-place quantization of nn_t
nn_error_t nn_quantize(nn_t *nn)
{
  if (!nn || nn->quantized) {
    return NN_ERROR_INVALID_ARGUMENT;
  }
  const int depth = (int)nn->depth;
  // Free the float-mode weight_scale/bias_scale placeholders (allocated by
  // nn_add_layer) before replacing them with the real quantized-mode arrays
  // below -- otherwise they leak.
  for (int L = 1; L < depth; L++)
    free(nn->weight_scale[L]);
  free(nn->weight_scale);
  free(nn->bias_scale);
  // Mark the network as quantized
  nn->quantized = true;
  // Allocate quantization arrays in the union fields
  nn->weight_quantized = malloc(depth * sizeof(int8_t *));
  nn->weight_scale = malloc(depth * sizeof(float *));
  nn->bias_quantized = malloc(depth * sizeof(int8_t *));
  nn->bias_scale = malloc(depth * sizeof(float));
  if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized || !nn->bias_scale) {
    return NN_ERROR_OUT_OF_MEMORY;
  }
  // Initialize layer 0 entries
  nn->weight_quantized[0] = NULL;
  nn->weight_scale[0] = NULL;
  nn->bias_quantized[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  // Quantize each layer > 1
  for (int L = 1; L < depth; L++) {
    if (nn->layer_type[L] == LAYER_TYPE_POOL) {
      // Pooling has no weights/bias to quantize
      nn->weight_quantized[L] = NULL;
      nn->weight_scale[L] = NULL;
      nn->bias_quantized[L] = NULL;
      nn->bias_scale[L] = 0.0f;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
    // Allocate per-layer arrays (one flat, row-major buffer for the weights)
    nn->weight_quantized[L] = malloc((size_t)rows * row_len * sizeof(int8_t));
    nn->weight_scale[L] = malloc(rows * sizeof(float));
    nn->bias_quantized[L] = malloc(bias_count * sizeof(int8_t));
    if (!nn->weight_quantized[L] || !nn->weight_scale[L] || !nn->bias_quantized[L]) {
      return NN_ERROR_OUT_OF_MEMORY;
    }
    // Compute one bias_scale for this layer
    float min_b = nn->bias[L][0], max_b = min_b;
    for (int i = 1; i < bias_count; i++) {
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
    // For each row (neuron or kernel) in this layer:
    for (int n = 0; n < rows; n++) {
      float *wrow = nn->weight[L] + n * row_len;
      int8_t *qrow = nn->weight_quantized[L] + n * row_len;
      // Compute per-row weight scale
      float min_w = wrow[0], max_w = min_w;
      for (int k = 1; k < row_len; k++) {
        if (wrow[k] < min_w)
          min_w = wrow[k];
        if (wrow[k] > max_w)
          max_w = wrow[k];
      }
      float row_scale = fmaxf(fabsf(min_w), fabsf(max_w)) / 127.0f;
      if (row_scale == 0.0f)
        row_scale = 1e-8f;
      nn->weight_scale[L][n] = row_scale;
      // Quantize each weight
      for (int k = 0; k < row_len; k++) {
        int8_t q = (int8_t)lroundf(wrow[k] / row_scale);
        qrow[k] = (q > 127 ? 127 : (q < -128 ? -128 : q));
      }
    }
    // Quantize the biases (bias_count of them, using the shared layer scale)
    for (int n = 0; n < bias_count; n++) {
      float borig = nn->bias[L][n];
      int8_t bq = (int8_t)lroundf(borig / layer_bias_scale);
      nn->bias_quantized[L][n] = (bq > 127 ? 127 : (bq < -128 ? -128 : bq));
    }
  }
  // Free all of the original float-side storage AFTER quantization
  for (int L = 1; L < depth; L++) {
    // weight/weight_adj are each one flat buffer per layer (or NULL for
    // POOL, where they were never allocated); free(NULL) is a no-op.
    free(nn->weight[L]);
    free(nn->weight_adj[L]);
    free(nn->bias[L]);
  }
  free(nn->weight);
  free(nn->weight_adj);
  free(nn->bias);
  nn->weight = NULL;
  nn->weight_adj = NULL;
  nn->bias = NULL;
  return NN_ERROR_NONE;
}

// Dequantize in-place: rebuild float weights/biases from the fixed-point model.
// Returns 0 on success, -1 on error.
nn_error_t nn_dequantize(nn_t *nn)
{
  if (!nn || !nn->quantized) {
    return NN_ERROR_INVALID_ARGUMENT;
  }
  const int depth = (int)nn->depth;
  // Allocate top-level float pointers
  nn->weight = malloc(depth * sizeof(*nn->weight));
  nn->weight_adj = malloc(depth * sizeof(*nn->weight_adj));
  nn->bias = malloc(depth * sizeof(*nn->bias));
  if (!nn->weight || !nn->weight_adj || !nn->bias) {
    return NN_ERROR_OUT_OF_MEMORY;
  }
  nn->weight[0] = NULL;
  nn->weight_adj[0] = NULL;
  nn->bias[0] = NULL;
  // For each layer >=1, rebuild float weight, weight_adj, bias
  for (int L = 1; L < depth; L++) {
    if (nn->layer_type[L] == LAYER_TYPE_POOL) {
      // Pooling has no weights/bias; nothing was quantized for it either.
      nn->weight[L] = NULL;
      nn->weight_adj[L] = NULL;
      nn->bias[L] = NULL;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
    // Allocate one flat, row-major float buffer per layer
    nn->weight[L] = malloc((size_t)rows * row_len * sizeof(*nn->weight[L]));
    nn->weight_adj[L] = malloc((size_t)rows * row_len * sizeof(*nn->weight_adj[L]));
    nn->bias[L] = malloc(bias_count * sizeof(*nn->bias[L]));
    if (!nn->weight[L] || !nn->weight_adj[L] || !nn->bias[L]) {
      return NN_ERROR_OUT_OF_MEMORY;
    }
    // Dequantize each weight; weight_adj was never meaningful in quantized
    // mode, so just zero it.
    for (int r = 0; r < rows; r++) {
      float wscale = nn->weight_scale[L][r];
      for (int c = 0; c < row_len; c++) {
        int idx = r * row_len + c;
        nn->weight[L][idx] = nn->weight_quantized[L][idx] * wscale;
        nn->weight_adj[L][idx] = 0.0f;
      }
    }
    // Dequantize the biases (bias_count of them, using the shared layer scale)
    for (int i = 0; i < bias_count; i++) {
      nn->bias[L][i] = nn->bias_quantized[L][i] * nn->bias_scale[L];
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
  nn->weight_quantized = NULL;
  nn->bias_quantized = NULL;
  // Reallocate fresh float-mode weight_scale/bias_scale placeholders,
  // matching what nn_add_layer sets up when a layer is first constructed --
  // without this, nn->weight_scale/bias_scale would be left dangling
  // (freed above, never reassigned), which nn_free() would later try to
  // read/free again.
  nn->weight_scale = malloc(depth * sizeof(*nn->weight_scale));
  nn->bias_scale = malloc(depth * sizeof(*nn->bias_scale));
  if (!nn->weight_scale || !nn->bias_scale) {
    return NN_ERROR_OUT_OF_MEMORY;
  }
  nn->weight_scale[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  for (int L = 1; L < depth; L++) {
    nn->bias_scale[L] = 0.0f;
    if (nn->layer_type[L] == LAYER_TYPE_POOL) {
      nn->weight_scale[L] = NULL;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
    nn->weight_scale[L] = malloc(rows * sizeof(*nn->weight_scale[L]));
    if (!nn->weight_scale[L]) {
      return NN_ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < rows; i++)
      nn->weight_scale[L][i] = 0.0f;
  }
  // Mark as float mode
  nn->quantized = false;
  return NN_ERROR_NONE;
}
