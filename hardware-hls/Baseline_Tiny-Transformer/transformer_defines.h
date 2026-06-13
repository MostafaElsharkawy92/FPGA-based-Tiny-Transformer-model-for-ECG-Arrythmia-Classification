/******************************************************************************
 * ECG Transformer - HLS Implementation
 * Description: Transformer model parameters and definitions for ECG classification
 *******************************************************************************/

#ifndef TRANSFORMER_DEFINES_H
#define TRANSFORMER_DEFINES_H

// Fixed-point arithmetic
//#include "ap_fixed.h"
//#define EXP_WIDTH 32
//#define CUSTOM_INT_WIDTH 8
//typedef ap_fixed<EXP_WIDTH, CUSTOM_INT_WIDTH> float24_t;
//typedef float float24_t;

// Model Configuration
#define NUM_CLASSES 5           // ['N', 'S', 'V', 'F', 'Q']
#define NUM_HEADS 8
#define HIDDEN_SIZE 2           // key_dim per head
#define EMBEDDING_DIM 16
#define MLP_DIM 128
#define KERNEL_SIZE 3
#define EMB_DEPTH 1
#define TRANSFORMER_LAYERS 1
#define HALF_WINDOW 99
#define BATCH_SIZE 1            // Inference mode, single sample

// Input dimensions
#define INPUT_LENGTH 198        // 2 * HALF_WINDOW
#define INPUT_CHANNELS 1
#define RR_FEATURES 2

// Patch embedding (Conv1D equivalent as Conv2D)
#define PATCH_CONV_KERNEL 3     // KERNEL_SIZE
#define PATCH_CONV_STRIDE 3     // KERNEL_SIZE
#define PATCH_CONV_FILTERS 16   // EMBEDDING_DIM

// After patch embedding: (198-3)/3+1 = 66 patches
#define NUM_PATCHES 66          // (INPUT_LENGTH - KERNEL_SIZE) / KERNEL_SIZE + 1
#define PATCH_DIM 16           // EMBEDDING_DIM

// Multi-Head Attention dimensions
#define HEAD_DIM 2             // HIDDEN_SIZE
#define TOTAL_HEAD_DIM 16      // NUM_HEADS * HEAD_DIM
#define QKV_DIM 48             // 3 * TOTAL_HEAD_DIM (Query, Key, Value)

// MLP (Feed-forward) dimensions
#define MLP_HIDDEN 128         // MLP_DIM
#define MLP_OUTPUT 16          // EMBEDDING_DIM

// RR feature embedding
#define RR_EMB_DIM 2

// Final classifier
#define CLASSIFIER_INPUT 18    // EMBEDDING_DIM + RR_EMB_DIM
#define CLASSIFIER_OUTPUT 5    // NUM_CLASSES

// Mathematical constants
#define SQRT_HEAD_DIM 1.414213562f  // sqrt(2.0) for HEAD_DIM=2
#define LAYER_NORM_EPS 1e-6f
#define GELU_COEFF 0.7978845608f    // sqrt(2/pi)

#endif // TRANSFORMER_DEFINES_H
