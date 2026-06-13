/******************************************************************************
 * ECG Transformer - Layer Function Declarations
 * Description: Header file with all layer function prototypes
 *******************************************************************************/
#ifndef TRANSFORMER_LAYERS_H
#define TRANSFORMER_LAYERS_H
#include "transformer_defines.h"
#include <hls_stream.h>



float gelu_approx(float x);

void add_residual_connection(hls::stream<float> &out,
                                    hls::stream<float> &processed_in,
                                    float residual_buffer[NUM_PATCHES][EMBEDDING_DIM]);

void save_residual_to_buffer(hls::stream<float> &stream_in,
                            hls::stream<float> &stream_out,
                            float residual_buffer[NUM_PATCHES][EMBEDDING_DIM]);
							
// Global average pooling
void global_average_pooling(hls::stream<float> &out, hls::stream<float> &in);

// Patch embedding using Conv2D (equivalent to Conv1D with stride=kernel_size)
void patch_embedding(hls::stream<float> &out, hls::stream<float> &in,
                           float weights[PATCH_CONV_KERNEL][1][1][PATCH_CONV_FILTERS],
                           float bias[PATCH_CONV_FILTERS]);

// Add positional embeddings
void add_position_embeddings(hls::stream<float> &out, hls::stream<float> &in,
                           float pos_emb[NUM_PATCHES][EMBEDDING_DIM]);
                           
// Dynamic Layer Normalization function
void layer_norm_stream(hls::stream<float> &out, hls::stream<float> &in,
                      float gamma[EMBEDDING_DIM], float beta[EMBEDDING_DIM]);
                      
// Multi-Head Attention          
void multi_head_attention(
    hls::stream<float> &out,
    hls::stream<float> &in,
    hls::stream<float> &temp_stream1, // Reusable stream
    hls::stream<float> &temp_stream2, // Reusable stream
    float qkv_weights[EMBEDDING_DIM][QKV_DIM],
    float qkv_bias[QKV_DIM],
    float out_weights[TOTAL_HEAD_DIM][EMBEDDING_DIM],
    float out_bias[EMBEDDING_DIM]);
	
// Feed-forward MLP block with GELU activation
void mlp_block(hls::stream<float> &out, hls::stream<float> &in,
               float fc1_weights[EMBEDDING_DIM][MLP_HIDDEN],
               float fc1_bias[MLP_HIDDEN],
               float fc2_weights[MLP_HIDDEN][EMBEDDING_DIM],
               float fc2_bias[EMBEDDING_DIM]);
               
                                                 
// RR feature embedding
void rr_embedding(hls::stream<float> &out, hls::stream<float> &in,
                  float weights[RR_FEATURES][RR_EMB_DIM]);

// Final classifier
void classifier(hls::stream<float> &out, hls::stream<float> &ecg_in, hls::stream<float> &rr_in,
               float weights[CLASSIFIER_INPUT][CLASSIFIER_OUTPUT],
               float bias[CLASSIFIER_OUTPUT]);
                     
#endif // TRANSFORMER_LAYERS_H