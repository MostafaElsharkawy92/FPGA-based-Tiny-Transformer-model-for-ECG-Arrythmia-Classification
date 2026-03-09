# Quick Start Guide

This guide provides a rapid walkthrough to get started with the FPGA-based Tiny Transformer project.

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] TensorFlow 2.x installed
- [ ] Xilinx Vitis HLS 2023.1+ (for hardware synthesis)
- [ ] Xilinx Vivado 2023.1+ (for FPGA deployment)
- [ ] Xilinx Kria KV260 board (for deployment)
- [ ] PYNQ image flashed on SD card

## 30-Minute Quick Start

### Option A: Software Only (Training & Evaluation)

**Time**: ~30-60 minutes (including training)

```bash
# 1. Install dependencies
pip install tensorflow numpy scikit-learn matplotlib seaborn wfdb scipy h5py pandas

# 2. Navigate to software directory
cd software/

# 3. Launch Jupyter
jupyter notebook "Baseline Tiny Transformer (Software).ipynb"

# 4. Run all cells (Runtime → Run All)
# Wait for training to complete (~30-60 min on GPU, longer on CPU)

# 5. Check outputs
ls -lh *.h5 *.json *.npz *.png
```

**Expected Results**:
- Test accuracy: ~98.83% (baseline)
- Model weights saved: `baseline_ecg_transformer.weights.h5`

---

### Option B: HLS Synthesis (Hardware Translation)

**Time**: ~20-40 minutes (synthesis)

**Prerequisites**: Vitis HLS 2023.1 installed

```bash
# 1. Navigate to HLS directory
cd hardware-hls/

# 2. Launch Vitis HLS
vitis_hls

# 3. In Vitis HLS GUI:
#    - Create New Project: "ecg_transformer_hls"
#    - Top Function: ecg_transformer
#    - Add files: transformer.cpp, transformer_layers.cpp, *.h
#    - Add testbench: transformer_testbench.cpp
#    - Device: xck26-sfvc784-2LV-c
#    - Clock: 13 ns (76.92 MHz)

# 4. Run C Simulation (verify correctness)
# 5. Run C Synthesis (generate RTL)
# 6. Run C/RTL Co-Simulation (verify RTL)
# 7. Export RTL → Vivado IP (.zip)
```

**Expected Results**:
- Latency: 56.62 ms
- Resource utilization: 30% LUTs, 37% BRAM, 27% DSP

---

### Option C: Full FPGA Deployment

**Time**: ~2-3 hours (implementation + bitstream generation)

**Prerequisites**: Vivado 2023.1, Kria KV260 with PYNQ

```bash
# 1. Create Vivado project
vivado

# 2. Create block design (see deployment/README.md)
#    - Add Zynq UltraScale+ MPSoC
#    - Add HLS IP (from Step B)
#    - Add AXI Interconnect & SmartConnect
#    - Make connections, assign addresses

# 3. Generate bitstream
#    - Synthesis (~15 min)
#    - Implementation (~25 min)
#    - Write Bitstream (~10 min)

# 4. Transfer to Kria KV260
scp design_1_wrapper.bit xilinx@<kria_ip>:~/
scp design_1_wrapper.hwh xilinx@<kria_ip>:~/

# 5. SSH to Kria and run inference
ssh xilinx@<kria_ip>
python3 ecg_inference.py
```

**Expected Results**:
- Inference latency: ~56.62 ms
- Classification output: Predicted class (N, S, V, F, Q)

---

## Common Issues

| Issue | Solution |
|-------|----------|
| **GPU out of memory during training** | Reduce `batch_size` to 64 or 32 in Config |
| **MIT-BIH download fails** | Manually download from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) |
| **HLS synthesis fails** | Check Vitis HLS version (2023.1+), verify device part |
| **Vivado bitstream error** | Validate block design, check address assignments |
| **PYNQ overlay load fails** | Verify .bit and .hwh files are in same directory |

---

## Directory Navigation

```
Start Here
    ↓
software/
    ├── README.md ← Read this first
    └── Baseline Tiny Transformer (Software).ipynb ← Train model
    ↓
hardware-hls/
    ├── README.md ← Read this
    ├── transformer.cpp ← Open in Vitis HLS
    └── ... (other .cpp/.h files)
    ↓
deployment/
    ├── README.md ← Deployment guide
    └── Vivado_BlockDiagram_Deployment through PYNQ.pdf ← Reference
```

---

## Verification Checklist

### Software (Step 1)
- [ ] Jupyter notebook runs without errors
- [ ] Training completes with validation accuracy > 95%
- [ ] Test accuracy ~98.83% (baseline)
- [ ] Model weights saved: `baseline_ecg_transformer.weights.h5`

### HLS (Step 2)
- [ ] C Simulation passes
- [ ] C Synthesis completes successfully
- [ ] C/RTL Co-Simulation passes
- [ ] IP exported as `.zip`

### FPGA (Step 3)
- [ ] Block design validates without errors
- [ ] Bitstream generated successfully
- [ ] PYNQ overlay loads on Kria
- [ ] Inference produces valid output

---

## Performance Targets

| Metric | Target | Actual (This Work) |
|--------|--------|--------------------|
| **Accuracy** | >98% | 98.83% (baseline) / 96.91% (dynamic) ✅ |
| **Latency** | <10 ms | 56.62 ms ⚠️ |
| **Parameters** | <10k | 6,643 (baseline) / 7,895 (dynamic) ✅ |
| **LUT Utilization** | <50% | 30% ✅ |
| **Power** | TBD | TBD |

**Status**: Functional baseline established. Optimization in progress (see paper Section IV).

---

## Next Steps After Quick Start

1. **Understand the baseline**: Read the paper in `docs/`
2. **Explore optimizations**: See `hardware-hls/README.md` for HLS directives
3. **Profile performance**: Use Vivado ILA or PYNQ profiling
4. **Compare with MCU**: Benchmark against GAP9 implementation
5. **Optimize**: Apply pipelining, parallelization, quantization

---

## Getting Help

- **Paper**: `docs/FPGA_Implementation_of_Tiny_Transformer_...pdf`
- **Presentation**: `docs/Presentation [Full].pptx`
- **Main README**: `README.md` (comprehensive guide)
- **Directory READMEs**: Detailed instructions in each folder

**Contact**: See `README.md` for author contact information.

---

**Ready to start?** Begin with `software/README.md`!
