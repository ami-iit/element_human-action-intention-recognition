#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV2D_KERNEL_0 {9, -49, 7, 52, -7, -82, -2, -2}

#define TENSOR_CONV2D_KERNEL_0_DEC_BITS {6}

#define TENSOR_CONV2D_BIAS_0 {-57, -51, -56, -37, -63, -50, -70, -67}

#define TENSOR_CONV2D_BIAS_0_DEC_BITS {7}

#define CONV2D_BIAS_LSHIFT {1}

#define CONV2D_OUTPUT_RSHIFT {6}

#define TENSOR_DENSE_KERNEL_0 {-63, -73, -54, -58, -59, -50, -79, -56, -6, -18, -35, -84, 16, 14, -70, -78, 12, 13, 3, -7, 8, -6, -3, -13, -8, -14, -9, 1, -13, -8, 12, -21, -68, -68, -82, -58, -59, -73, -93, -57, 11, -81, -31, 44, -3, -46, -36, 41, -21, 7, -5, 52, -43, -4, -14, 27, 6, -81, -5, 6, -43, -51, -35, 23, -64, -77, -89, -46, -88, -55, -67, -79, 15, 22, -34, -85, 19, 9, -28, -40, -7, -8, 18, -4, 11, -20, -12, -16, 20, -16, -17, -12, 9, 5, -7, -19, -53, -72, -94, -63, -65, -35, -62, -49, 14, -47, -31, 9, -39, -58, -78, 3, -12, 13, -16, 36, -34, -3, -21, 16, -24, -84, 7, 2, -19, -45, -20, 31, -80, -58, -75, -37, -83, -68, -49, -42, 5, 2, -9, -54, 11, -4, -34, -39, -7, 18, 11, -19, -14, 18, 17, -12, -7, 1, 16, 18, 6, -9, 23, 6, -72, -80, -69, -47, -66, -44, -63, -53, -2, -74, -18, -3, -5, -43, -48, -7, 0, 31, 4, 23, -3, 15, -29, 21, 13, -66, 7, 15, -11, -86, -26, 0, -89, -74, -93, -48, -63, -55, -58, -34, 23, 22, -43, -55, 12, -5, -56, -59, 24, -9, -13, 23, 17, -5, -23, 3, 9, 19, 17, 13, -11, 10, 7, -20, -66, -59, -65, -36, -77, -42, -60, -44, 6, -51, -13, -11, -23, -69, -31, 30, 8, 15, -1, 32, -43, 10, -36, 4, 19, -49, 23, -1, -36, -66, -38, 2, -66, -68, -92, -40, -87, -40, -88, -70, -4, 3, -32, -76, 8, -19, -51, -68, 22, -8, 14, -5, 22, 13, 18, -18, -1, -3, -1, 22, 16, -1, 13, 4, -93, -51, -82, -63, -93, -62, -85, -75, 0, -38, -15, 42, -13, -56, -58, 37, -18, 30, 1, 28, -15, 15, -33, 5, -1, -65, -1, 57, -47, -71, -39, -2, -75, -50, -68, -43, -62, -38, -72, -51, 4, -9, -16, -47, 9, 22, -62, -65, -22, -21, -3, 22, -18, 7, 0, -17, -18, -16, 0, 18, -1, -5, 8, -16, -88, -63, -83, -79, -58, -50, -66, -73, -12, -62, -31, -6, 11, -41, -66, 0, -22, 26, 6, 47, -45, 2, -7, 6, -5, -46, 16, 41, -46, -72, -39, 0, -78, -45, -60, -77, -54, -72, -50, -73, -2, -5, -48, -46, -13, -18, -44, -66, 17, -15, -2, -17, 10, -15, -5, -12, 17, 11, 9, -11, -10, -22, 15, -2, -65, -44, -75, -44, -82, -70, -53, -47, -17, -88, -38, 38, -36, -55, -48, 36, -13, 26, -7, 15, -31, -6, -13, 22, 3, -44, 1, 12, -22, -64, -44, 27, -62, -72, -72, -58, -87, -60, -85, -66, 3, 3, -19, -50, -18, 22, -52, -70, 14, -4, 7, 22, 20, 9, 18, 14, 8, 6, -16, -2, 4, -6, -24, 17, -54, -39, -64, -35, -74, -79, -76, -76, 5, -69, -38, 26, -10, -44, -33, 9, 11, 32, 17, 8, -14, 29, -7, 11, 17, -85, -2, 41, -46, -70, -34, 15, -68, -66, -78, -73, -56, -53, -92, -32, 9, -1, -15, -61, -18, -17, -43, -64, 23, -21, 19, -6, -23, -10, 1, -6, 21, -19, -7, 8, -20, -21, -7, -20, -57, -55, -86, -54, -72, -45, -74, -38, 11, -82, -37, 34, -9, -44, -34, 25, 11, 3, 22, 15, -12, 16, -11, 15, -5, -84, -13, -8, -5, -58, -10, -2, -87, -58, -66, -54, -53, -46, -85, -61, -3, -3, -23, -78, -17, 8, -35, -74, 14, 11, -17, 14, -24, -18, -10, 9, 5, -12, -3, -2, -20, 6, 1, 8, -56, -71, -56, -38, -52, -66, -51, -75, 12, -41, -38, 35, -15, -68, -35, 18, -1, 49, -6, 17, -47, 33, -16, 9, -6, -71, 7, 32, -9, -68, -16, 32, -73, -72, -33, -41, -70, -87, -83, -67, 16, -20, 11, -94, 2, 4, -62, -6, -19, 2, 3, 23, 22, 12, -6, -20, 0, -18, 11, 7, -2, 23, 14, 13, -29, -30, -36, -68, -59, -60, -57, -71, -56, -56, 107, -10, -18, -38, -61, -23, 16, 31, 8, 46, -34, -42, -35, -47, -52, -41, 1, 15, -12, -51, -21, -8, -72, -32, -43, -57, -77, -89, -75, -52, -7, 23, 16, -85, -8, 15, -48, -11, -16, -18, 14, -12, -6, -13, 7, -12, -6, -9, -7, -11, -16, 13, -20, 6, -58, -50, -45, -60, -78, -70, -94, -65, -61, -70, 80, -5, 18, -19, -30, 11, 18, 37, 31, 47, -3, -37, -30, -24, -81, -63, 38, 24, -1, -32, -22, -10, -41, -53, -48, -66, -72, -77, -78, -81, 19, -9, 16, -56, 15, 0, -48, 5, 15, -7, -10, -16, -4, -16, -10, 8, 22, -19, 9, -14, 19, 18, 1, 1, -32, -65, -40, -60, -61, -62, -81, -92, -85, -64, 109, -39, 9, -36, -29, 1, 14, 13, 35, 39, 13, -32, -13, -23, -65, -68, 9, 32, -11, -49, -20, -14, -60, -64, -60, -54, -76, -80, -87, -51, 11, -23, -1, -62, 4, 7, -61, 13, -1, 13, -1, 10, 14, 22, -12, 4, 20, 22, -7, 22, -3, 12, 18, 2, -66, -59, -28, -54, -61, -95, -89, -79, -59, -50, 104, -9, 0, -31, -35, -22, 50, 39, 17, 26, -45, -50, -8, -45, -77, -61, 22, 38, -30, -51, -10, -28, -45, -31, -52, -61, -49, -60, -50, -51, -10, -2, 6, -93, 3, 21, -51, -7, 14, 7, -1, -15, 21, -12, -23, 12, 15, -23, 24, 14, 16, 23, 14, 1, -67, -43, -61, -35, -68, -66, -88, -62, -47, -45, 109, -37, -9, -44, -48, -4, 22, 27, 65, 3, -39, -21, -13, -61, -61, -79, 40, 18, -53, -64, -30, -39, -33, -44, -65, -54, -60, -48, -68, -57, 10, 2, 7, -94, 10, -23, -28, 17, -14, 23, -16, 15, -9, -21, 1, -15, 13, -8, -1, 8, 14, -1, -16, 14, -42, -45, -37, -76, -63, -59, -68, -89, -60, -44, 126, -2, -32, -2, -31, -3, 2, 8, 44, -4, -22, -38, -3, -23, -73, -58, 14, -5, -32, -35, -20, -26, -71, -39, -60, -66, -90, -53, -95, -92, -3, -17, 10, -89, -7, 22, -47, -22, -20, -1, 2, 17, 7, -21, 2, -8, 8, 4, -5, -14, 24, -12, 20, -22, -36, -32, -61, -64, -76, -54, -53, -52, -45, -42, 119, 5, 7, -13, -66, -2, 6, 38, 35, 7, -27, -30, -8, -30, -44, -73, 29, 36, -3, -63, -13, -24, -35, -64, -54, -59, -70, -75, -67, -85, -15, -7, 9, -88, -4, -19, -44, -19, -4, 8, -24, 8, 11, -8, -15, 10, -18, 5, 8, 16, -11, 8, 8, 21, -26, -61, -60, -50, -65, -50, -56, -74, -42, -53, 103, -36, -26, -23, -31, 8, 46, 23, 58, 0, -45, -47, -52, -46, -55, -48, 41, 27, -44, -32, -26, -41, -74, -71, -32, -50, -80, -67, -51, -58, 9, 2, 8, -64, -21, -16, -53, -4, 9, -1, -5, -8, -9, 4, 18, 23, 5, 23, -10, 9, 13, -10, -8, 15, -39, -55, -31, -37, -77, -93, -74, -62, -65, -43, 105, -4, 4, -8, -49, 14, 47, -6, 51, 29, -39, -35, -9, -33, -47, -53, 15, -4, -19, -21, -24, -17, -64, -66, -50, -48, -89, -90, -64, -52, 10, -19, 11, -67, -21, 6, -41, -5, 0, -23, -18, -2, -20, -20, 4, -10, -10, -8, 23, 12, -10, 3, 19, -22, -37, -39, -25, -41, -63, -77, -88, -58, -72, -65, 104, -13, -30, -17, -62, 13, 31, -9, 38, 3, -8, -26, -25, -36, -54, -68, 44, 24, -16, -47, -21, -10, -76, 0, -95, 12, -6, -23, 19, 7, -60, -27, -82, -2, -44, -49, -50, -40, -77, 4, -61, -23, 9, -11, 8, -16, -68, -20, -54, -36, -49, -61, -62, -55, -50, -21, -73, -15, -3, -24, -5, 13, -87, -47, -81, 0, -39, -38, -49, -43, -70, -22, -73, -1, -9, 17, -16, 1, -72, -52, -72, -9, -23, -53, -46, -38, -55, -19, -73, -6, -12, 7, 5, 9, -83, -26, -78, -41, -25, -50, -61, -46, -68, 16, -50, 2, -19, 14, -11, 2, -89, -56, -49, -38, -69, -74, -41, -60, -72, -14, -59, 2, 22, 24, 14, 20, -76, -18, -60, -41, -48, -66, -49, -32, -58, 24, -47, 11, -19, 22, 18, -11, -80, -51, -90, -21, -56, -75, -52, -61, -74, 7, -70, -16, -24, 7, 9, 15, -70, -25, -82, -23, -38, -68, -57, -70, -88, -7, -81, -21, 19, -11, -9, -4, -85, -53, -53, -25, -55, -55, -31, -24, -38, -16, -43, -31, -21, 5, -8, -7, -72, -69, -70, 60, 44, -80, 43, 48, -49, 2, -43, -36, 15, -2, -9, 6, -68, -74, -56, 54, 57, -41, 34, 35, -44, 20, -48, -81, 1, 12, 8, 10, -76, -46, -33, 30, 54, -58, 34, 69, -46, -21, -34, -3, 20, 10, 13, 6, -70, -45, -77, 46, 63, -48, 44, 47, -57, 17, -34, -41, 12, -7, 8, -12, -36, -79, -54, 33, 63, -75, 39, 33, -61, -14, -37, -73, -10, -12, 11, 8, -40, -41, -46, 34, 33, -37, 46, 54, -31, 12, -42, -40, 8, 10, -15, -13, -29, -67, -74, 34, 40, -38, 34, 67, -62, -18, -72, -78, 0, 24, -14, -23, -42, -79, -77, 60, 50, -69, 72, 40, -54, -17, -43, -34, -10, 0, 16, 20, -60, -71, -63, 46, 41, -57, 36, 55, -38, 21, -72, -47, -1, 10, -22, 7, -50, -63, -50, 55, 35, -70, 54, 52}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS {7}

#define TENSOR_DENSE_BIAS_0 {-71, -58, -71, -56, -46, -53, -76, -71, -71, -54}

#define TENSOR_DENSE_BIAS_0_DEC_BITS {7}

#define DENSE_BIAS_LSHIFT {2}

#define DENSE_OUTPUT_RSHIFT {6}

#define TENSOR_DENSE_1_KERNEL_0 {92, 33, 93, -90, 33, -66, 48, 95, 38, 2}

#define TENSOR_DENSE_1_KERNEL_0_DEC_BITS {8}

#define TENSOR_DENSE_1_BIAS_0 {77}

#define TENSOR_DENSE_1_BIAS_0_DEC_BITS {8}

#define DENSE_1_BIAS_LSHIFT {3}

#define DENSE_1_OUTPUT_RSHIFT {5}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC 2
#define INPUT_1_OUTPUT_OFFSET 7
#define CONV2D_OUTPUT_DEC 2
#define CONV2D_OUTPUT_OFFSET 1
#define MAX_POOLING2D_OUTPUT_DEC 2
#define MAX_POOLING2D_OUTPUT_OFFSET 0
#define FLATTEN_OUTPUT_DEC 2
#define FLATTEN_OUTPUT_OFFSET 0
#define DENSE_OUTPUT_DEC 3
#define DENSE_OUTPUT_OFFSET 10
#define DENSE_1_OUTPUT_DEC 6
#define DENSE_1_OUTPUT_OFFSET 109

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data[20] = {0};

const nnom_shape_data_t tensor_input_1_0_dim[] = {10, 2, 1};
const nnom_qformat_param_t tensor_input_1_0_dec[] = {2};
const nnom_qformat_param_t tensor_input_1_0_offset[] = {0};
const nnom_tensor_t tensor_input_1_0 = {
    .p_data = (void*)nnom_input_data,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0
};
const int8_t tensor_conv2d_kernel_0_data[] = TENSOR_CONV2D_KERNEL_0;

const nnom_shape_data_t tensor_conv2d_kernel_0_dim[] = {1, 1, 1, 8};
const nnom_qformat_param_t tensor_conv2d_kernel_0_dec[] = TENSOR_CONV2D_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv2d_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv2d_kernel_0 = {
    .p_data = (void*)tensor_conv2d_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv2d_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv2d_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv2d_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 4,
    .bitwidth = 8
};
const int8_t tensor_conv2d_bias_0_data[] = TENSOR_CONV2D_BIAS_0;

const nnom_shape_data_t tensor_conv2d_bias_0_dim[] = {8};
const nnom_qformat_param_t tensor_conv2d_bias_0_dec[] = TENSOR_CONV2D_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv2d_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv2d_bias_0 = {
    .p_data = (void*)tensor_conv2d_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv2d_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv2d_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv2d_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv2d_output_shift[] = CONV2D_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv2d_bias_shift[] = CONV2D_BIAS_LSHIFT;
const nnom_conv2d_config_t conv2d_config = {
    .super = {.name = "conv2d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv2d_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv2d_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv2d_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv2d_bias_shift, 
    .filter_size = 8,
    .kernel_size = {1, 1},
    .stride_size = {1, 1},
    .padding_size = {0, 0},
    .dilation_size = {1, 1},
    .padding_type = PADDING_VALID
};

const nnom_pool_config_t max_pooling2d_config = {
    .super = {.name = "max_pooling2d"},
    .padding_type = PADDING_SAME,
    .output_shift = 0,
    .kernel_size = {1, 1},
    .stride_size = {1, 1},
    .num_dim = 2
};

const nnom_flatten_config_t flatten_config = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data[] = TENSOR_DENSE_KERNEL_0;

const nnom_shape_data_t tensor_dense_kernel_0_dim[] = {160, 10};
const nnom_qformat_param_t tensor_dense_kernel_0_dec[] = TENSOR_DENSE_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_dense_kernel_0 = {
    .p_data = (void*)tensor_dense_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data[] = TENSOR_DENSE_BIAS_0;

const nnom_shape_data_t tensor_dense_bias_0_dim[] = {10};
const nnom_qformat_param_t tensor_dense_bias_0_dec[] = TENSOR_DENSE_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_bias_0_offset[] = {0};
const nnom_tensor_t tensor_dense_bias_0 = {
    .p_data = (void*)tensor_dense_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift[] = DENSE_OUTPUT_RSHIFT;
const nnom_qformat_param_t dense_bias_shift[] = DENSE_BIAS_LSHIFT;
const nnom_dense_config_t dense_config = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift
};
const int8_t tensor_dense_1_kernel_0_data[] = TENSOR_DENSE_1_KERNEL_0;

const nnom_shape_data_t tensor_dense_1_kernel_0_dim[] = {10, 1};
const nnom_qformat_param_t tensor_dense_1_kernel_0_dec[] = TENSOR_DENSE_1_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_1_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_dense_1_kernel_0 = {
    .p_data = (void*)tensor_dense_1_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_1_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_1_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_1_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_1_bias_0_data[] = TENSOR_DENSE_1_BIAS_0;

const nnom_shape_data_t tensor_dense_1_bias_0_dim[] = {1};
const nnom_qformat_param_t tensor_dense_1_bias_0_dec[] = TENSOR_DENSE_1_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_1_bias_0_offset[] = {0};
const nnom_tensor_t tensor_dense_1_bias_0 = {
    .p_data = (void*)tensor_dense_1_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_1_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_1_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_1_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_1_output_shift[] = DENSE_1_OUTPUT_RSHIFT;
const nnom_qformat_param_t dense_1_bias_shift[] = DENSE_1_BIAS_LSHIFT;
const nnom_dense_config_t dense_1_config = {
    .super = {.name = "dense_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_1_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_dense_1_bias_0,
    .output_shift = (nnom_qformat_param_t *)&dense_1_output_shift,
    .bias_shift = (nnom_qformat_param_t *)&dense_1_bias_shift
};
static int8_t nnom_output_data[1] = {0};

const nnom_shape_data_t tensor_output_dim[] = {1};
const nnom_qformat_param_t tensor_output_dec[] = {DENSE_1_OUTPUT_DEC};
const nnom_qformat_param_t tensor_output_offset[] = {0};
const nnom_tensor_t tensor_output = {
    .p_data = (void*)nnom_output_data,
    .dim = (nnom_shape_data_t*)tensor_output_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_output_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_output_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output_config = {
    .super = {.name = "output"},
    .tensor = (nnom_tensor_t*)&tensor_output
};
/* model version */
#define NNOM_MODEL_VERSION (10000*0 + 100*4 + 0)

/* nnom model */
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[7];

	check_model_version(NNOM_MODEL_VERSION);
	new_model(&model);

	layer[0] = input_s(&input_1_config);
	layer[1] = model.hook(conv2d_s(&conv2d_config), layer[0]);
	layer[2] = model.hook(maxpool_s(&max_pooling2d_config), layer[1]);
	layer[3] = model.hook(flatten_s(&flatten_config), layer[2]);
	layer[4] = model.hook(dense_s(&dense_config), layer[3]);
	layer[5] = model.hook(dense_s(&dense_1_config), layer[4]);
	layer[6] = model.hook(output_s(&output_config), layer[5]);
	model_compile(&model, layer[0], layer[6]);
	return &model;
}
