/******************************************************************************
 * ECG Transformer - Layer Implementations
 *******************************************************************************/

#include <hls_math.h>
#include "transformer_layers.h"
#include "transformer_defines.h"

float gelu_approximate(float x) {
	return x * (1.0f / (1.0f + std::exp(-1.702f * x)));
}

void add_residual_connection(hls::stream<float> &out,
                                    hls::stream<float> &processed_in,
                                    float residual_buffer[NUM_PATCHES][EMBEDDING_DIM]) {

    RESIDUAL_LOOP: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            float processed_val = processed_in.read();
            float result = processed_val + residual_buffer[patch][dim];
            out.write(result);
        }
    }
}

void save_residual_to_buffer(hls::stream<float> &stream_in,
                            hls::stream<float> &stream_out,
                            float residual_buffer[NUM_PATCHES][EMBEDDING_DIM]) {


    SAVE_LOOP: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            float val = stream_in.read();
            residual_buffer[patch][dim] = val;  // Save for later residual
            stream_out.write(val);              // Pass through to next stage
        }
    }
}

void global_average_pooling(hls::stream<float> &out, hls::stream<float> &in) {
    static float sums[EMBEDDING_DIM];

    // Initialize sums
    INIT_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
        sums[dim] = 0.0f;
    }

    // Accumulate all patches
    PATCH_LOOP: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        DIM_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            float val = in.read();
            sums[dim] += val;
        }
    }

    // Average and output
    OUTPUT_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
        out.write(sums[dim] / NUM_PATCHES);
    }
}

void patch_embedding(hls::stream<float> &out, hls::stream<float> &in,
                    float weights[PATCH_CONV_KERNEL][1][1][PATCH_CONV_FILTERS],
                    float bias[PATCH_CONV_FILTERS]) {

    static float sample_buffer[INPUT_LENGTH];

    // Read input stream
    READ_INPUT: for (int i = 0; i < INPUT_LENGTH; i++) {
        sample_buffer[i] = in.read();
    }

    // Process and output
    PATCH_LOOP: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        int start_idx = patch * PATCH_CONV_STRIDE;

        FILTER_LOOP: for (int filter = 0; filter < PATCH_CONV_FILTERS; filter++) {
            float conv_sum = bias[filter];

            KERNEL_LOOP: for (int k = 0; k < PATCH_CONV_KERNEL; k++) {
                int sample_idx = start_idx + k;
                if (sample_idx < INPUT_LENGTH) {
                    conv_sum += sample_buffer[sample_idx] * weights[k][0][0][filter];
                }
            }

            out.write(conv_sum);
        }
    }
}

void add_position_embeddings(hls::stream<float> &out, hls::stream<float> &in,
                            float pos_emb[NUM_PATCHES][EMBEDDING_DIM]) {
    PATCH_LOOP: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        DIM_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            float patch_val = in.read();
            out.write(patch_val + pos_emb[patch][dim]);
        }
    }
}

void layer_norm_stream(hls::stream<float> &out, hls::stream<float> &in,
                       float gamma[EMBEDDING_DIM], float beta[EMBEDDING_DIM]) {

    // Process each patch sequentially
    PATCH_LOOP: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        static float data[EMBEDDING_DIM];
        //#pragma HLS ARRAY_PARTITION variable=data complete dim=1

        // Read patch data
        READ_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
           // #pragma HLS UNROLL
            data[dim] = in.read();
        }

        // Compute mean
        float mean = 0.0f;
        MEAN_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            //#pragma HLS UNROLL
            mean += data[dim];
        }
        mean /= EMBEDDING_DIM;

        // Compute variance
        float variance = 0.0f;
        VAR_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            //#pragma HLS UNROLL
            float diff = data[dim] - mean;
            variance += diff * diff;
        }
        variance /= EMBEDDING_DIM;
        float inv_std = 1.0f / hls::sqrt(variance + LAYER_NORM_EPS);

        // Write normalized output
        NORM_LOOP: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            //#pragma HLS UNROLL
            float normalized = (data[dim] - mean) * inv_std;
            out.write(gamma[dim] * normalized + beta[dim]);
        }
    }
}

void multi_head_attention(
    hls::stream<float> &out,
    hls::stream<float> &in,
    hls::stream<float> &temp_stream1,
    hls::stream<float> &temp_stream2,
    float qkv_weights[EMBEDDING_DIM][QKV_DIM],
    float qkv_bias[QKV_DIM],
    float out_weights[TOTAL_HEAD_DIM][EMBEDDING_DIM],
    float out_bias[EMBEDDING_DIM]) {

    // Reduced static arrays - only essential buffers
    static float qkv_buffer[NUM_PATCHES][QKV_DIM];
    //#pragma HLS ARRAY_PARTITION variable=qkv_buffer cyclic factor=4 dim=2

    // Small buffers for head processing
    static float current_head_q[NUM_PATCHES][HEAD_DIM];
    static float current_head_k[NUM_PATCHES][HEAD_DIM];
    static float current_head_v[NUM_PATCHES][HEAD_DIM];
    //#pragma HLS ARRAY_PARTITION variable=current_head_q complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=current_head_k complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=current_head_v complete dim=2

    // Attention computation buffers - reused per head
    static float attention_scores[NUM_PATCHES][NUM_PATCHES];
    static float attention_weights[NUM_PATCHES][NUM_PATCHES];
    //#pragma HLS ARRAY_PARTITION variable=attention_scores complete dim=2
    //#pragma HLS ARRAY_PARTITION variable=attention_weights complete dim=2

    // Output accumulator - much smaller than before
    static float head_outputs[NUM_PATCHES][TOTAL_HEAD_DIM];
    //#pragma HLS ARRAY_PARTITION variable=head_outputs cyclic factor=4 dim=2

    // PHASE 1: QKV COMPUTATION - Direct from stream
    QKV_COMPUTATION: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        // Read input patch directly from stream
        static float input_patch[EMBEDDING_DIM];
      //  #pragma HLS ARRAY_PARTITION variable=input_patch complete dim=1

        for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
        //    #pragma HLS PIPELINE II=1
            input_patch[dim] = in.read();
        }

        // Compute QKV for this patch
        for (int qkv_dim = 0; qkv_dim < QKV_DIM; qkv_dim++) {
          //  #pragma HLS PIPELINE II=1
           // #pragma HLS UNROLL factor=4
            float sum = qkv_bias[qkv_dim];

            for (int emb_dim = 0; emb_dim < EMBEDDING_DIM; emb_dim++) {
             //   #pragma HLS UNROLL factor=8
                sum += input_patch[emb_dim] * qkv_weights[emb_dim][qkv_dim];
            }
            qkv_buffer[patch][qkv_dim] = sum;
        }
    }

    // Initialize output
    INIT_OUTPUT: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        for (int dim = 0; dim < TOTAL_HEAD_DIM; dim++) {
            //#pragma HLS UNROLL factor=4
            head_outputs[patch][dim] = 0.0f;
        }
    }

    // PHASE 2: Process each head sequentially
    HEAD_PROCESSING: for (int head = 0; head < NUM_HEADS; head++) {
        int head_offset = head * HEAD_DIM;

        // Extract current head's Q, K, V
        EXTRACT_HEAD_QKV: for (int patch = 0; patch < NUM_PATCHES; patch++) {
            for (int dim = 0; dim < HEAD_DIM; dim++) {
              //  #pragma HLS UNROLL
                current_head_q[patch][dim] = qkv_buffer[patch][head_offset + dim];
                current_head_k[patch][dim] = qkv_buffer[patch][head_offset + dim + TOTAL_HEAD_DIM];
                current_head_v[patch][dim] = qkv_buffer[patch][head_offset + dim + 2 * TOTAL_HEAD_DIM];
            }
        }

        // Compute attention scores
        ATTENTION_SCORES: for (int q_patch = 0; q_patch < NUM_PATCHES; q_patch++) {
            for (int k_patch = 0; k_patch < NUM_PATCHES; k_patch++) {
                //#pragma HLS PIPELINE II=1
                float dot_product = 0.0f;

                for (int dim = 0; dim < HEAD_DIM; dim++) {
                  //  #pragma HLS UNROLL
                    dot_product += current_head_q[q_patch][dim] * current_head_k[k_patch][dim];
                }
                attention_scores[q_patch][k_patch] = dot_product * (1.0f / SQRT_HEAD_DIM);
            }
        }

        // Softmax computation (same as before)
        SOFTMAX: for (int q_patch = 0; q_patch < NUM_PATCHES; q_patch++) {
            float max_score = attention_scores[q_patch][0];
            for (int k_patch = 1; k_patch < NUM_PATCHES; k_patch++) {
                //#pragma HLS UNROLL
                if (attention_scores[q_patch][k_patch] > max_score) {
                    max_score = attention_scores[q_patch][k_patch];
                }
            }

            float sum_exp = 0.0f;
            static float exp_scores[NUM_PATCHES];
            for (int k_patch = 0; k_patch < NUM_PATCHES; k_patch++) {
                //#pragma HLS UNROLL
                float exp_val = hls::exp(attention_scores[q_patch][k_patch] - max_score);
                exp_scores[k_patch] = exp_val;
                sum_exp += exp_val;
            }

            for (int k_patch = 0; k_patch < NUM_PATCHES; k_patch++) {
                //#pragma HLS UNROLL
                attention_weights[q_patch][k_patch] = exp_scores[k_patch] / sum_exp;
            }
        }

        // Apply attention to values and accumulate in output
        APPLY_ATTENTION: for (int q_patch = 0; q_patch < NUM_PATCHES; q_patch++) {
            for (int dim = 0; dim < HEAD_DIM; dim++) {
                //#pragma HLS PIPELINE II=1
                float weighted_sum = 0.0f;

                for (int k_patch = 0; k_patch < NUM_PATCHES; k_patch++) {
                  //  #pragma HLS UNROLL
                    weighted_sum += attention_weights[q_patch][k_patch] * current_head_v[k_patch][dim];
                }

                head_outputs[q_patch][head_offset + dim] = weighted_sum;
            }
        }
    }

    // PHASE 3: Output projection using temp streams
    // First, write to temp_stream1
    WRITE_TO_TEMP: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        for (int dim = 0; dim < TOTAL_HEAD_DIM; dim++) {
            //#pragma HLS PIPELINE II=1
            temp_stream1.write(head_outputs[patch][dim]);
        }
    }

    // Read from temp_stream1 and compute projection to temp_stream2
    OUTPUT_PROJECTION: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        static float head_data[TOTAL_HEAD_DIM];
        //#pragma HLS ARRAY_PARTITION variable=head_data cyclic factor=4 dim=1

        // Read head output
        for (int dim = 0; dim < TOTAL_HEAD_DIM; dim++) {
          //  #pragma HLS PIPELINE II=1
            head_data[dim] = temp_stream1.read();
        }

        // Compute projection
        for (int emb_dim = 0; emb_dim < EMBEDDING_DIM; emb_dim++) {
            //#pragma HLS PIPELINE II=1
            float sum = out_bias[emb_dim];

            for (int head_dim = 0; head_dim < TOTAL_HEAD_DIM; head_dim++) {
              //  #pragma HLS UNROLL factor=4
                sum += head_data[head_dim] * out_weights[head_dim][emb_dim];
            }
            temp_stream2.write(sum);
        }
    }

    // Finally, copy from temp_stream2 to output
    FINAL_OUTPUT: for (int patch = 0; patch < NUM_PATCHES; patch++) {
        for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
            //#pragma HLS PIPELINE II=1
            out.write(temp_stream2.read());
        }
    }
}

void mlp_block(hls::stream<float> &out, hls::stream<float> &in,
               float fc1_weights[EMBEDDING_DIM][MLP_HIDDEN],
               float fc1_bias[MLP_HIDDEN],
               float fc2_weights[MLP_HIDDEN][EMBEDDING_DIM],
               float fc2_bias[EMBEDDING_DIM]) {

    static float input_vec[EMBEDDING_DIM];
    static float hidden_vec[MLP_HIDDEN];
    static float output_vec[EMBEDDING_DIM];
   // #pragma HLS ARRAY_PARTITION variable=input_vec complete dim=1
   // #pragma HLS ARRAY_PARTITION variable=hidden_vec cyclic factor=8 dim=1
   // #pragma HLS ARRAY_PARTITION variable=output_vec complete dim=1

    // Process one patch at a time
    PATCH_LOOP: for (int patch = 0; patch < NUM_PATCHES; patch++) {

        // Read input patch
        READ_INPUT: for (int dim = 0; dim < EMBEDDING_DIM; dim++) {
     //       #pragma HLS UNROLL
            input_vec[dim] = in.read();
        }

        // FC1 computation
        FC1_LOOP: for (int h = 0; h < MLP_HIDDEN; h++) {
       //     #pragma HLS PIPELINE II=1
        //    #pragma HLS UNROLL factor=8
           float sum = fc1_bias[h];
            for (int d = 0; d < EMBEDDING_DIM; d++) {
          //      #pragma HLS UNROLL
                sum += input_vec[d] * fc1_weights[d][h];
            }
            hidden_vec[h] = gelu_approximate(sum);
        }

        // FC2 computation
        FC2_LOOP: for (int d = 0; d < EMBEDDING_DIM; d++) {
            //#pragma HLS PIPELINE II=1
           // #pragma HLS UNROLL off
            float sum = fc2_bias[d];
            for (int h = 0; h < MLP_HIDDEN; h++) {
             //   #pragma HLS UNROLL factor=8
                sum += hidden_vec[h] * fc2_weights[h][d];
            }
            output_vec[d] = gelu_approximate(sum);
        }

        // Write outputs
        WRITE_OUT: for (int d = 0; d < EMBEDDING_DIM; d++) {
           // #pragma HLS UNROLL
            out.write(output_vec[d]);
        }
    }
}

void rr_embedding(hls::stream<float> &out, hls::stream<float> &in,
                  float weights[RR_FEATURES][RR_EMB_DIM]) {

    static float input_data[RR_FEATURES];
   // #pragma HLS ARRAY_PARTITION variable=input_data complete dim=1

    // Read input
    READ_RR: for (int i = 0; i < RR_FEATURES; i++) {
     //   #pragma HLS UNROLL
        input_data[i] = in.read();
    }

    // Compute embedding
    EMBED_LOOP: for (int out_dim = 0; out_dim < RR_EMB_DIM; out_dim++) {
       // #pragma HLS UNROLL
        float sum = 0.0f;

        for (int in_dim = 0; in_dim < RR_FEATURES; in_dim++) {
         //   #pragma HLS UNROLL
            sum += input_data[in_dim] * weights[in_dim][out_dim];
        }
        out.write(sum);
    }
}

void classifier(hls::stream<float> &out, hls::stream<float> &ecg_in, hls::stream<float> &rr_in,
                float weights[CLASSIFIER_INPUT][CLASSIFIER_OUTPUT],
                float bias[CLASSIFIER_OUTPUT]) {

    static float combined_input[CLASSIFIER_INPUT];
    //#pragma HLS ARRAY_PARTITION variable=combined_input complete dim=1

    // Read ECG features
    READ_ECG: for (int i = 0; i < EMBEDDING_DIM; i++) {
      //  #pragma HLS UNROLL
        combined_input[i] = ecg_in.read();
    }

    // Read RR features
    READ_RR: for (int i = 0; i < RR_EMB_DIM; i++) {
        //#pragma HLS UNROLL
        combined_input[EMBEDDING_DIM + i] = rr_in.read();
    }

    // Final classification
    CLASS_LOOP: for (int out_class = 0; out_class < CLASSIFIER_OUTPUT; out_class++) {
       // #pragma HLS UNROLL
        float sum = bias[out_class];

        for (int in_dim = 0; in_dim < CLASSIFIER_INPUT; in_dim++) {
         //   #pragma HLS UNROLL
            sum += combined_input[in_dim] * weights[in_dim][out_class];
        }
        out.write(sum);
    }
}

