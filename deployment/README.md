# FPGA Deployment on Xilinx Kria KV260

This directory contains the deployment guide and block diagram for integrating the HLS-generated ECG Transformer accelerator on the Kria KV260 platform using PYNQ.

## Contents

| File | Description |
|------|-------------|
| `Vivado_BlockDiagram_Deployment through PYNQ.pdf` | Complete Vivado block design schematic |

## System Architecture

### Hardware Platform

**Xilinx Kria KV260 Vision AI Starter Kit**
- **FPGA**: Zynq UltraScale+ MPSoC XCZU5EV
- **Processing System (PS)**: Quad-core ARM Cortex-A53 @ 1.2 GHz
- **Programmable Logic (PL)**:
  - 117K LUTs
  - 288 BRAM (18Kb)
  - 1,200 DSP slices
  - 234K Flip-Flops
- **Memory**: 4GB DDR4
- **Interfaces**: Gigabit Ethernet, USB, DisplayPort, microSD

### Block Design Components

Refer to `Vivado_BlockDiagram_Deployment through PYNQ.pdf` for the visual schematic.

#### 1. Zynq UltraScale+ MPSoC (`zynq_ultra_ps_e_0`)

**Processing System Configuration**:
- **Master AXI Interfaces** (PS → PL):
  - `M_AXI_HPM0_FPD`: Control path to accelerator (AXI4-Lite, 32-bit)
  - `M_AXI_HPM1_FPD`: Additional master interface (if needed)

- **Slave AXI Interfaces** (PL → PS):
  - `S_AXI_HP0_FPD`: High-performance port 0 (AXI4, 128-bit)
  - `S_AXI_HP1_FPD`: High-performance port 1 (optional)

- **Clocks**:
  - `pl_clk0`: 100 MHz (system clock)
  - `pl_clk1`: 76.92 MHz (accelerator clock)

- **Interrupts**:
  - `pl_ps_irq0[0:0]`: Accelerator completion interrupt

- **Reset**:
  - `pl_resetn0`: Active-low reset

#### 2. ECG Transformer Accelerator (`ecg_transformer_top_0`)

**Interfaces**:
- `s_axi_control`: AXI4-Lite slave for register access
  - Start/stop control
  - Status registers
  - Pointer to DDR memory buffers

- `m_axi_gmem0`: AXI4 master for ECG input data
- `m_axi_gmem1`: AXI4 master for RR interval data
- `m_axi_gmem2`: AXI4 master for output data

- `ap_clk`: Clock input (76.92 MHz)
- `ap_rst_n`: Active-low reset
- `interrupt`: Completion signal to PS

#### 3. AXI Interconnect (`ps8_0_axi_periph`)

**Purpose**: Connect PS master interfaces to accelerator control interface

**Configuration**:
- **Inputs**:
  - `S00_AXI`: From PS `M_AXI_HPM0_FPD`
  - `S01_AXI`: (optional additional master)

- **Outputs**:
  - `M00_AXI`: To accelerator `s_axi_control`

- **Clock/Reset**:
  - `ACLK`: 100 MHz
  - `ARESETN`: Synchronized reset

#### 4. AXI SmartConnect (`smartconnect_0`)

**Purpose**: High-bandwidth data path between accelerator and DDR memory

**Configuration**:
- **Inputs** (from accelerator):
  - `S00_AXI`: Connected to `m_axi_gmem0` (ECG data)
  - `S01_AXI`: Connected to `m_axi_gmem1` (RR data)
  - `S02_AXI`: Connected to `m_axi_gmem2` (output)

- **Outputs** (to PS):
  - `M00_AXI`: To PS `S_AXI_HP0_FPD` (128-bit DDR access)
  - `M01_AXI`: (optional, to `S_AXI_HP1_FPD`)

- **Features**:
  - Automatic data width conversion
  - Out-of-order transaction support
  - Multiple outstanding transactions

#### 5. Processor System Reset (`rst_ps8_0_96M`)

**Purpose**: Generate synchronized resets for all clock domains

**Inputs**:
- `slowest_sync_clk`: 100 MHz
- `ext_reset_in`: From PS `pl_resetn0`

**Outputs**:
- `peripheral_aresetn[0:0]`: Synchronized reset for peripherals
- `interconnect_aresetn[0:0]`: Reset for interconnects

## Vivado Project Setup

### Step 1: Create Project

```tcl
# Launch Vivado 2023.1
vivado

# Create new project
create_project ecg_transformer_project ./vivado_project -part xck26-sfvc784-2LV-c
```

Or use GUI:
- **File → New Project**
- **Board**: Kria KV260 Vision AI Starter Kit
- **Part**: `xck26-sfvc784-2LV-c`

### Step 2: Create Block Design

```tcl
# Create block design
create_bd_design "design_1"

# Add Zynq UltraScale+ MPSoC
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0
```

**Configure Zynq PS** (in GUI):
1. Double-click `zynq_ultra_ps_e_0`
2. **PS-PL Configuration**:
   - Enable `M_AXI_HPM0_FPD` (32-bit, AXI4-Lite compatible)
   - Enable `S_AXI_HP0_FPD` (128-bit)
3. **Clock Configuration**:
   - PL Fabric Clock 0: 100 MHz (enabled)
   - PL Fabric Clock 1: 76.92 MHz (enabled)
4. **Interrupts**:
   - Enable `pl_ps_irq0[0:0]`

### Step 3: Add Custom HLS IP

```tcl
# Add IP repository (HLS-generated IP)
set_property ip_repo_paths {./hls_ip/ecg_transformer_top.zip} [current_project]
update_ip_catalog

# Add custom IP to block design
create_bd_cell -type ip -vlnv xilinx.com:hls:ecg_transformer_top:1.0 ecg_transformer_top_0
```

### Step 4: Add Interconnects

**AXI Interconnect** (control path):
```tcl
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps8_0_axi_periph
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] [get_bd_cells ps8_0_axi_periph]
```

**AXI SmartConnect** (data path):
```tcl
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_0]
```

### Step 5: Make Connections

**Automated connections**:
```tcl
# Run connection automation
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]

# Connect AXI interfaces
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins ps8_0_axi_periph/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ps8_0_axi_periph/M00_AXI] [get_bd_intf_pins ecg_transformer_top_0/s_axi_control]

# Connect SmartConnect
connect_bd_intf_net [get_bd_intf_pins ecg_transformer_top_0/m_axi_gmem0] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ecg_transformer_top_0/m_axi_gmem1] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins ecg_transformer_top_0/m_axi_gmem2] [get_bd_intf_pins smartconnect_0/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

# Connect interrupt
connect_bd_net [get_bd_pins ecg_transformer_top_0/interrupt] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]
```

**Manual connections** (if needed):
- Clock: Connect all `aclk` ports to `pl_clk0` (100 MHz) or `pl_clk1` (76.92 MHz) as appropriate
- Reset: Connect `aresetn` ports to `rst_ps8_0_96M/peripheral_aresetn`

### Step 6: Assign Addresses

```tcl
# Automatic address assignment
assign_bd_address

# Verify addresses
# Control registers should be in PS memory map (e.g., 0xA000_0000)
```

**Typical Address Map**:
```
0xA000_0000 - 0xA000_FFFF : ecg_transformer_top_0 control registers
0x0000_0000 - 0x7FFF_FFFF : DDR memory (accessible by accelerator via SmartConnect)
```

### Step 7: Validate and Generate

```tcl
# Validate design
validate_bd_design

# Generate HDL wrapper
make_wrapper -files [get_files design_1.bd] -top

# Add wrapper to project
add_files -norecurse ./vivado_project/ecg_transformer_project.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
```

## Implementation Flow

### Synthesis

```tcl
launch_runs synth_1 -jobs 8
wait_on_run synth_1
```

**Expected**: ~10-20 minutes

### Implementation

```tcl
launch_runs impl_1 -to_phase route_design -jobs 8
wait_on_run impl_1
```

**Expected**: ~20-30 minutes

### Generate Bitstream

```tcl
launch_runs impl_1 -to_phase write_bitstream -jobs 8
wait_on_run impl_1
```

**Outputs**:
- `design_1_wrapper.bit` - FPGA bitstream
- `design_1_wrapper.hwh` - Hardware handoff file (for PYNQ)

## PYNQ Deployment

### Prepare Kria KV260

1. **Flash PYNQ Image**:
   - Download PYNQ 3.0+ for Kria KV260
   - Flash to microSD card (16GB+)
   - Insert SD card and boot

2. **Network Setup**:
   - Connect Kria to network via Ethernet
   - Find IP address: `ifconfig` or check router
   - Default credentials: `xilinx` / `xilinx`

3. **Transfer Files**:
```bash
# From your PC
scp design_1_wrapper.bit xilinx@<kria_ip>:~/
scp design_1_wrapper.hwh xilinx@<kria_ip>:~/
```

### Python Control Script

Create `ecg_inference.py` on Kria:

```python
from pynq import Overlay, allocate
import numpy as np
import time

# Load overlay
print("Loading bitstream...")
overlay = Overlay("design_1_wrapper.bit")

# Get accelerator handle
ecg_accel = overlay.ecg_transformer_top_0

# Check register map
print("Accelerator registers:")
print(ecg_accel.register_map)

# Allocate physically contiguous buffers
ecg_buffer = allocate(shape=(198,), dtype=np.float32)
rr_buffer = allocate(shape=(2,), dtype=np.float32)
output_buffer = allocate(shape=(5,), dtype=np.float32)

# Load test ECG data (example: synthetic normal beat)
ecg_buffer[:] = np.sin(2 * np.pi * 1.2 * np.arange(198) / 360)  # Placeholder
rr_buffer[:] = [0.8, 0.85]  # Pre-RR, Post-RR

# Synchronize to device
ecg_buffer.sync_to_device()
rr_buffer.sync_to_device()

# Configure accelerator
ecg_accel.register_map.ecg_data = ecg_buffer.physical_address
ecg_accel.register_map.rr_data = rr_buffer.physical_address
ecg_accel.register_map.output_data = output_buffer.physical_address

# Start inference
print("Starting inference...")
start_time = time.time()
ecg_accel.register_map.CTRL.AP_START = 1

# Wait for completion (polling)
while ecg_accel.register_map.CTRL.AP_DONE == 0:
    pass

end_time = time.time()
latency_ms = (end_time - start_time) * 1000

# Read results
output_buffer.sync_from_device()
logits = output_buffer.copy()

# Apply softmax
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)

# Predict class
class_names = ['N', 'S', 'V', 'F', 'Q']
predicted_class = np.argmax(probs)

print(f"\nResults:")
print(f"Latency: {latency_ms:.2f} ms")
print(f"Predicted class: {class_names[predicted_class]}")
print(f"Probabilities: {probs}")
print(f"  N (Normal):     {probs[0]:.4f}")
print(f"  S (Supra-vent): {probs[1]:.4f}")
print(f"  V (Vent):       {probs[2]:.4f}")
print(f"  F (Fusion):     {probs[3]:.4f}")
print(f"  Q (Unknown):    {probs[4]:.4f}")

# Cleanup
ecg_buffer.freebuffer()
rr_buffer.freebuffer()
output_buffer.freebuffer()
```

**Run**:
```bash
sudo python3 ecg_inference.py
```

**Expected Output**:
```
Loading bitstream...
Accelerator registers: {...}
Starting inference...

Results:
Latency: 56.62 ms
Predicted class: N
Probabilities: [0.9823 0.0102 0.0051 0.0018 0.0006]
  N (Normal):     0.9823
  S (Supra-vent): 0.0102
  V (Vent):       0.0051
  F (Fusion):     0.0018
  Q (Unknown):    0.0006
```

## Performance Measurement

### Latency Profiling

```python
import time

num_runs = 100
latencies = []

for i in range(num_runs):
    start = time.perf_counter()

    # Configure and start
    ecg_accel.register_map.CTRL.AP_START = 1
    while ecg_accel.register_map.CTRL.AP_DONE == 0:
        pass

    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # ms

print(f"Average latency: {np.mean(latencies):.2f} ms")
print(f"Std deviation:   {np.std(latencies):.2f} ms")
print(f"Min latency:     {np.min(latencies):.2f} ms")
print(f"Max latency:     {np.max(latencies):.2f} ms")
```

### Resource Utilization

Check Vivado implementation reports:

```bash
# On development PC
vivado -mode batch -source report_utilization.tcl
```

`report_utilization.tcl`:
```tcl
open_run impl_1
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
report_power -file power.rpt
```

## Troubleshooting

### Issue: Bitstream loading fails
**Solution**:
- Check file permissions: `chmod 644 *.bit *.hwh`
- Verify PYNQ version compatibility

### Issue: Accelerator hangs
**Solution**:
- Check AXI address ranges don't overlap
- Verify clock connections (all synchronous)
- Increase AXI timeout in SmartConnect properties

### Issue: Incorrect results
**Solution**:
- Verify HLS co-simulation passed
- Check data alignment (32-bit aligned addresses)
- Validate endianness (ARM is little-endian)

### Issue: Low performance
**Solution**:
- Confirm clock frequency: `cat /sys/kernel/debug/clk/clk_summary`
- Check AXI bandwidth with ILA (Integrated Logic Analyzer)
- Profile memory access patterns

## Advanced: Interrupt-Based Control

Replace polling with interrupt handling:

```python
from pynq import GPIO

# Get interrupt pin
interrupt = GPIO(GPIO.get_gpio_pin(0), 'in')

def wait_for_interrupt(timeout=1.0):
    import select
    fds = [interrupt._impl.fileno()]
    readable, _, _ = select.select(fds, [], [], timeout)
    return len(readable) > 0

# Start accelerator
ecg_accel.register_map.CTRL.AP_START = 1

# Wait for interrupt
if wait_for_interrupt():
    print("Inference completed (interrupt)")
else:
    print("Timeout!")
```

## Next Steps

- **Optimization**: Apply HLS optimizations from `2_Hardware_HLS/README.md`
- **Integration**: Connect to real-time ECG sensor input
- **Power Measurement**: Use Xilinx Power Estimator or on-board sensors
- **ASIC Tape-out**: Migrate to open-source PDK for affordable healthcare

---

**Status**: ✅ **Functional** - Deployment successful, performance optimization ongoing.
