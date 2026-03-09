# Hardware HLS Implementation

This directory contains the Vitis HLS C++ implementation of the Tiny Transformer for FPGA acceleration.

## Files Overview

| File | Description |
|------|-------------|
| `transformer_defines.h` | Model parameters, dimensions, and macros |
| `transformer_layers.h` | Layer function declarations |
| `transformer_layers.cpp` | Layer implementations (attention, FFN, normalization) |
| `transformer.cpp` | Top-level function with embedded weights |
| `transformer_testbench.cpp` | C/RTL co-simulation testbench |

## Architecture

### Model Specifications

```cpp
// From transformer_defines.h
#define NUM_CLASSES 5           // N, S, V, F, Q
#define NUM_HEADS 8             // Multi-head attention
#define HIDDEN_SIZE 2           // Key dimension per head
#define EMBEDDING_DIM 16        // Patch embedding dimension
#define MLP_DIM 128            // Feed-forward hidden dimension
#define INPUT_LENGTH 198        // ECG samples
#define NUM_PATCHES 66          // (198-3)/3+1
#define RR_FEATURES 2          // Pre-RR, Post-RR intervals
```

### Data Flow

```
ecg_data[198] + rr_data[2]
    ↓
load_to_stream
    ↓
patch_embedding (Conv1D) → 66×16
    ↓
positional_encoding (add)
    ↓
layer_normalization
    ↓
multi_head_attention (8 heads)
    ↓
residual_add
    ↓
layer_normalization
    ↓
feed_forward_network (128→16)
    ↓
residual_add
    ↓
global_average_pooling → 1×16
    ↓
concat with RR embedding → 18
    ↓
classification_layer → 5 logits
    ↓
output_stream
```

## Key Implementation Details

### 1. Memory Management

HLS streams are used for data flow:

```cpp
hls::stream<float> main_data_stream("main_data");
#pragma HLS stream variable=main_data_stream depth=1056

hls::stream<float> temp_stream1("temp1");
#pragma HLS stream variable=temp_stream1 depth=1056

hls::stream<float> rr_stream("rr_data");
#pragma HLS stream variable=rr_stream depth=32
```

**Stream Depths**:
- `main_data_stream`: 1056 (66 patches × 16 dim)
- `temp_stream1/2`: 1056 (intermediate activations)
- `rr_stream`: 32 (RR features)

### 2. Embedded Weights

All trained weights from the software model are embedded as static arrays:

```cpp
static float patch_conv_weights[3][1][1][16] = { ... };
static float patch_conv_bias[16] = { ... };
static float pos_embeddings[66][16] = { ... };
static float attention_qkv_weights[16][48] = { ... };
static float mlp_weights_1[16][128] = { ... };
static float mlp_weights_2[128][16] = { ... };
static float classifier_weights[18][5] = { ... };
// ... and more
```

### 3. Layer Normalization

Implemented with numerical stability:

```cpp
void layer_normalization(
    hls::stream<float> &input,
    hls::stream<float> &output,
    const float gamma[EMBEDDING_DIM],
    const float beta[EMBEDDING_DIM],
    int seq_len
) {
    float eps = LAYER_NORM_EPS;  // 1e-6

    // Compute mean and variance
    // Apply normalization: (x - mean) / sqrt(var + eps)
    // Scale and shift: gamma * norm + beta
}
```

### 4. Multi-Head Self-Attention

Currently implemented sequentially (optimization opportunity):

```cpp
void multi_head_attention(
    hls::stream<float> &input,
    hls::stream<float> &output,
    // weights...
) {
    // 1. Linear projections: Q, K, V
    // 2. Split into heads: 8 heads × (66 seq × 2 dim)
    // 3. Scaled dot-product attention per head
    //    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    // 4. Concatenate heads
    // 5. Output projection
}
```

**Key Challenge**: Heads processed sequentially → major bottleneck (see Optimizations).

### 5. Feed-Forward Network

Two-layer MLP with GELU activation:

```cpp
void feed_forward_network(
    hls::stream<float> &input,
    hls::stream<float> &output,
    const float weights1[EMBEDDING_DIM][MLP_DIM],
    const float bias1[MLP_DIM],
    const float weights2[MLP_DIM][EMBEDDING_DIM],
    const float bias2[EMBEDDING_DIM]
) {
    // Layer 1: 16 → 128 with GELU
    // Layer 2: 128 → 16
}
```

**GELU Approximation**:
```cpp
float gelu(float x) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
}
```

## Vitis HLS Project Setup

### Step 1: Create Project

```bash
vitis_hls
```

In GUI:
1. **File → New Project**
2. **Project name**: `ecg_transformer_hls`
3. **Location**: Your workspace
4. **Top Function**: `ecg_transformer`
5. **Clock Period**: 13 ns (76.92 MHz)

### Step 2: Add Files

**Design Files** (in order):
- `transformer_defines.h`
- `transformer_layers.h`
- `transformer_layers.cpp`
- `transformer.cpp`

**Testbench Files**:
- `transformer_testbench.cpp`

### Step 3: Set Device

**Part Selection**:
- **Family**: Zynq UltraScale+
- **Device**: `xck26-sfvc784-2LV-c`
- **Package**: sfvc784
- **Speed Grade**: -2LV

This corresponds to the **Xilinx Kria KV260** platform.

### Step 4: Configure Solution

**Clock Settings**:
- **Target clock**: 13 ns (76.92 MHz)
- **Uncertainty**: 1.3 ns (10%)

**Interface Settings** (in `transformer.cpp`):
```cpp
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=ecg_data
#pragma HLS INTERFACE s_axilite port=rr_data
```

## Synthesis Flow

### C Simulation

Verify functional correctness:

```bash
# In Vitis HLS GUI
Project → Run C Simulation
```

**Expected**: Test passes, output matches reference.

### C Synthesis

Generate RTL:

```bash
Solution → Run C Synthesis → Active Solution
```

**Synthesis Report**:
```
================================================================
== Performance Estimates
================================================================
+ Timing:
    * Summary:
    +---------+--------+--------+--------+--------+
    |  Clock  | Target | Estimated | Uncertainty | Slack |
    +---------+--------+--------+--------+--------+
    | ap_clk  | 13.00  | 10.876  | 1.30   | 0.82  |
    +---------+--------+--------+--------+--------+

+ Latency (clock cycles):
    * Summary:
    +---------+---------+-----------+-----------+
    |  Latency (min/max/avg)       | Interval  |
    +---------+---------+-----------+-----------+
    | 4355291 | 4355291 | 4355291   | 4355292   |
    +---------+---------+-----------+-----------+

+ Latency (absolute time):
    * 56.62 ms @ 76.92 MHz

================================================================
== Utilization Estimates
================================================================
+ Summary:
    +-----------------+-------+-------+--------+-------+
    |      Name       | BRAM  | DSP   | FF     | LUT   |
    +-----------------+-------+-------+--------+-------+
    | Expression      | -     | 342   | 47415  | 35881 |
    +-----------------+-------+-------+--------+-------+
    | Total           | 109   | 342   | 47415  | 35881 |
    +-----------------+-------+-------+--------+-------+
    | Available       | 288   | 1200  | 234240 | 117120|
    +-----------------+-------+-------+--------+-------+
    | Utilization (%) | 37    | 27    | 20     | 30    |
    +-----------------+-------+-------+--------+-------+
```

### C/RTL Co-Simulation

Verify RTL behavior:

```bash
Solution → Run C/RTL Cosimulation
```

**Options**:
- **RTL**: Verilog or VHDL
- **Dump Trace**: waveform (optional, large file)

**Expected**: Cosim passes, outputs match C simulation.

### Export RTL

Export IP for Vivado integration:

```bash
Solution → Export RTL
```

**Format**: Vivado IP (.zip)

**Output**: `ecg_transformer_top.zip`

## Optimization Directives

### Current Implementation

```cpp
#pragma HLS INTERFACE axis port=output
#pragma HLS pipeline off
#pragma HLS stream variable=main_data_stream depth=1056
```

**Status**: Minimal optimization, primarily sequential execution.

### Recommended Optimizations

#### 1. Parallel Attention Heads

```cpp
void multi_head_attention(...) {
    #pragma HLS DATAFLOW  // Enable parallel head processing

    for (int h = 0; h < NUM_HEADS; h++) {
        #pragma HLS UNROLL  // Unroll head loop
        // Process head h
    }
}
```

**Expected**: 8× speedup in attention layer.

#### 2. Array Partitioning

```cpp
static float qkv_weights[16][48];
#pragma HLS ARRAY_PARTITION variable=qkv_weights cyclic factor=8 dim=2
```

**Benefit**: Parallel memory access for multiple heads.

#### 3. Loop Pipelining

```cpp
QKV_COMPUTATION:
for (int i = 0; i < NUM_PATCHES; i++) {
    #pragma HLS PIPELINE II=1
    // Compute Q, K, V projections
}
```

**Benefit**: Overlapped computation, reduced latency.

#### 4. Function Inlining

```cpp
#pragma HLS INLINE recursive
```

**Benefit**: Eliminates function call overhead, exposes optimization opportunities.

## Performance Analysis

### Current Bottlenecks

From synthesis report:

| Loop/Function | Iterations | Cycles | Bottleneck |
|---------------|-----------|--------|------------|
| `QKV_COMPUTATION` | 6,307 | ~23.5 ms | Sequential head processing |
| `HEAD_PROCESSING` | 2,362 | - | Memory access patterns |
| `FFN_LAYER1` | 36,261 | ~31.1 ms | **Major bottleneck** |

**Feed-Forward Network (54.9% of total latency)**:
- 36,261 iterations suggest inefficient loop structure
- Opportunities for loop tiling and memory optimization

### Optimization Roadmap

1. **Phase 1** (Quick wins):
   - Add `PIPELINE` directives to critical loops
   - Array partitioning for weights
   - **Expected**: 2-3× speedup

2. **Phase 2** (Architectural):
   - Parallel attention heads with `DATAFLOW`
   - Memory access reordering
   - **Expected**: 5-8× speedup

3. **Phase 3** (Advanced):
   - Quantization (INT8)
   - Layer fusion
   - **Expected**: 10-15× speedup (target: 4-6 ms latency)

## Testbench

`transformer_testbench.cpp` provides:

1. **Golden reference**: Expected outputs from software model
2. **Test cases**: Sample ECG inputs with known labels
3. **Tolerance checking**: Floating-point comparison with epsilon

**Usage**:
```cpp
int main() {
    float ecg_input[198] = { /* test ECG */ };
    float rr_input[2] = {0.8, 0.85};
    float output[5];

    // Run HLS function
    ecg_transformer(output, ecg_input, rr_input);

    // Compare with golden reference
    float expected[5] = { /* from software */ };
    for (int i = 0; i < 5; i++) {
        assert(abs(output[i] - expected[i]) < 1e-3);
    }

    return 0;
}
```

## Troubleshooting

### Issue: Synthesis fails with resource overflow
**Solution**: Reduce unroll factors or array partition sizes

### Issue: Long synthesis time
**Solution**: Disable waveform dumping, reduce test vectors

### Issue: Co-sim mismatch
**Solution**: Check data types (float precision), verify weight embedding

## Next Steps

After successful synthesis and co-simulation:
1. Export RTL IP
2. Proceed to `3_FPGA-Kria KV260-Deployment/`
3. Integrate IP in Vivado block design
4. Generate bitstream and deploy on Kria KV260

---

**Optimization Status**: ⚠️ **Room for improvement** - Current implementation is functional but not optimized. See paper Section III for detailed performance analysis.
