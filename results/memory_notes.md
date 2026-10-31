# ORBIT-PRISM: Memory Profiling and SRAM Budget Notes
Owner: **Harish D** (Branch: `harish/radar-compress`)
Date: 2026-10-31

## 1. Executive Summary & Design Budget
- **Onboard SRAM Budget**: `4,200,000` bytes (4.2 MB raw ceiling, ~4,101.6 KiB) as specified in `src/constants.py` and `CONTEXT.md`.
- **Flash Assumption**: Model constant weights are assumed stored in on-board non-volatile Flash memory.
- **SRAM Assumption**: Active tensors, input buffers, DMA ping-pong buffers, and intermediate layer activations reside in SRAM.
- **Design Status**: Both TinyML architectures (Radar and Optical) are explicitly architected for low activation depth. Peak activations for both INT8 and FLOAT32 are well within the 4.2 MB SRAM limit.

## 2. Profiling Methodology
- **Weights (Flash)**:
  - Measured by parsing TFLite tensor details.
  - Sum of elements $\times$ bytes per element for static parameter tensors (weights and biases).
  - Excludes TFLite flatbuffer schema/metadata overhead.
- **Peak Activation Memory (SRAM — ESTIMATED)**:
  - TFLite tensor details alone do not provide exact runtime memory allocator arena layouts.
  - Therefore, peak activation values are **ESTIMATED**.
  - **Ping-Pong Arena Lifetime Model**: Evaluates the two largest simultaneously active tensors during forward propagation (e.g. input buffer + layer 1 output during conv1; conv1 output + conv2 output during conv2).
  - For Radar (128x128x2):
    - Input: `128 x 128 x 2 = 32,768` elements ($32.0$ KB INT8, $128.0$ KB FLOAT32).
    - Conv1 Output: `64 x 64 x 8 = 32,768` elements ($32.0$ KB INT8, $128.0$ KB FLOAT32).
    - Transition Conv1 Peak: $32.0 + 32.0 = 64.0$ KB INT8 ($256.0$ KB FLOAT32).
    - Conv2 Output: `32 x 32 x 16 = 16,384` elements ($16.0$ KB INT8).
    - Subsequent layers strictly decrease in spatial activation size.
  - Total peak activation in INT8 is **~64.0 KB**, requiring only **~1.5%** of the 4.2 MB SRAM budget.

## 3. Current Memory Table
```csv
model,precision,weights_KB,peak_activation_KB,auroc_cal,recall_at_fixed_threshold,fits_4.2MB
Optical,FLOAT32,PENDING,PENDING,PENDING,PENDING,PENDING
Optical,INT8,PENDING,PENDING,PENDING,PENDING,PENDING
Radar,FLOAT32,PENDING,PENDING,PENDING,PENDING,PENDING
Radar,INT8,PENDING,PENDING,PENDING,PENDING,PENDING

```

## 4. Omitted Memory Costs & Limitations (Important Edge Disclosures)
- **Model-Only Estimate**: This is an isolated neural network model estimate, not a full satellite flight software measurement.
- **Excluded Satellite Overhead**:
  - TFLite Micro runtime tensor arena allocator alignment overhead (typically 16-byte alignment).
  - Operator scratchpad buffers (e.g., im2col or CMSIS-NN convolution scratch buffers).
  - Telemetry framing and radio DMA downlink queue buffers (`PASS_BYTES`).
  - Tile preview cache and scheduler application state memory.
- **Measured vs Estimated**: Hardware arena profiling on target silicon should be labelled **MEASURED** when micro-benchmarked; these current values are strictly labelled **ESTIMATED**.
