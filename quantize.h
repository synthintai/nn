#ifndef QUANTIZE_H
#define QUANTIZE_H

typedef enum {              
    QUANTIZATION_METHOD_SYMMETRIC,
    QUANTIZATION_METHOD_ASYMMETRIC,
    QUANTIZATION_METHOD_PER_CHANNEL
} quantization_method_t;

typedef struct {
    quantization_method_t method;
    int bit_depth;
    float global_scale;
    float min_value;
    float max_value;
} quantization_config_t;

typedef struct {        
    float mean_absolute_error;
    float max_absolute_error;
    float root_mean_square_error;
} quantization_error_t;

// Function declarations
void print_usage();
nn_quantized_t* nn_quantize(nn_t* network, quantization_method_t method, int bit_depth);
int nn_save_quantized(nn_quantized_t* quantized_network, const char* path);
float* nn_predict_quantized(nn_quantized_t* qmodel, float* input);
quantization_error_t nn_analyze_quantization_error(nn_t* original_network, nn_quantized_t* quantized_network);

#endif /* QUANTIZE_H */ 
