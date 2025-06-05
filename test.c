/*
 * Neural Network test program
 * Adjusted to use nn_predict for both floating‐point and quantized models.
 *
 * Copyright (c) 2019-2025 SynthInt Technologies, LLC
 * https://synthint.ai
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "nn.h"
#include "data_prep.h"

int main(void)
{
    nn_t    *model;
    data_t  *data;
    float   *prediction;
    int      num_samples;
    int      correct;
    int      true_positive;
    int      false_positive;

    //----------------------------------------------------------------
    // 1) Load a previously saved model.  The model.txt file could
    //    represent either a floating‐point model or a quantized model.
    //----------------------------------------------------------------
    model = nn_load_model("model.txt");
    if (model == NULL) {
        printf("Error: Missing or invalid model file.\n");
        return 1;
    }

    //----------------------------------------------------------------
    // 2) Load training data into memory
    //----------------------------------------------------------------
    data = data_load("train.csv",
                     model->width[0],                  // input dimension
                     model->width[model->depth - 1]);  // output dimension
    if (data == NULL) {
        printf("Error: Could not load training data.\n");
        nn_free(model);
        return 1;
    }

    //----------------------------------------------------------------
    // 3) Evaluate accuracy on the training set
    //----------------------------------------------------------------
    num_samples = 0;
    correct     = 0;
    for (int i = 0; i < data->num_rows; i++) {
        num_samples++;

        // Call nn_predict() unconditionally, regardless of quantized or not.
        prediction = nn_predict(model, data->input[i]);

        true_positive  = 0;
        false_positive = 0;

        // Binary decision: target >= 0.5 → positive class, else negative
        for (int j = 0; j < model->width[model->depth - 1]; j++) {
            if (data->target[i][j] >= 0.5f) {
                if (prediction[j] >= 0.5f) {
                    true_positive++;
                }
            } else {
                if (prediction[j] >= 0.5f) {
                    false_positive++;
                }
            }
        }

        // We consider the sample correct if exactly one output neuron
        // is “on” for a true‐positive case and none for false‐positive.
        if ((true_positive == 1) && (false_positive == 0)) {
            correct++;
        }

        // Note: nn_predict returns either:
        //   - a pointer into model->neuron[...] for floating‐point models, or
        //   - a newly allocated float array for quantized models
        // In either case, we do NOT free() here.  The caller of nn_predict
        // must manage any internal or allocated buffers.  (Per the updated API,
        // nn_predict should handle memory lifetime itself.)
    }

    printf("Train: %d/%d = %2.2f%%\n",
           correct, num_samples,
           (correct * 100.0f) / (float)num_samples);

    data_free(data);

    //----------------------------------------------------------------
    // 4) Load unseen (test) data and evaluate
    //----------------------------------------------------------------
    data = data_load("test.csv",
                     model->width[0],
                     model->width[model->depth - 1]);
    if (data == NULL) {
        printf("Error: Could not load test data.\n");
        nn_free(model);
        return 1;
    }

    num_samples = 0;
    correct     = 0;
    for (int i = 0; i < data->num_rows; i++) {
        num_samples++;

        prediction = nn_predict(model, data->input[i]);

        true_positive  = 0;
        false_positive = 0;
        for (int j = 0; j < model->width[model->depth - 1]; j++) {
            if (data->target[i][j] >= 0.5f) {
                if (prediction[j] >= 0.5f) {
                    true_positive++;
                }
            } else {
                if (prediction[j] >= 0.5f) {
                    false_positive++;
                }
            }
        }
        if ((true_positive == 1) && (false_positive == 0)) {
            correct++;
        }

        // Again, do not free prediction here; nn_predict manages its own buffers.
    }

    printf("Test : %d/%d = %2.2f%%\n",
           correct, num_samples,
           (correct * 100.0f) / (float)num_samples);

    data_free(data);

    //----------------------------------------------------------------
    // 5) Clean up
    //----------------------------------------------------------------
    nn_free(model);
    return 0;
}
