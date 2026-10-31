# ORBIT-PRISM: Memory Profiling and SRAM Budget Notes
Owner: **Harish D** (Branch: `harish/radar-compress`)

## 1. Executive Summary & Design Budget
- **Budget**: `4,200,000` bytes (4.2 MB, decimal; about 4,101.6 KiB) as specified in `src/constants.py` and `CONTEXT.md`.
- **Accounting (conservative)**: `total_bytes = weights_bytes + peak_activation_bytes + tile_buffer_bytes`. Weights are counted against the budget as well as in flash, and the uint8 tile staged from flash is counted in addition to the arena (which also holds the model's own input tensor).
- **Status**: every row is marked in the `basis` column. `untrained_architecture` rows are valid for sizes (they depend on the architecture, not on training) but `recall` and `ece` are `NOT_MEASURED` until a trained model is evaluated.

## 2. Profiling Methodology
- **Weights**: sum of elements x bytes per element over static tensors in the TFLite file; excludes flatbuffer schema overhead.
- **Peak activation memory (ESTIMATED)**: TFLite tensor details do not give the runtime allocator layout, so the peak is the sum of the two largest dynamic tensors alive together (a ping-pong model). It is an estimate, not a measurement on target silicon.
- **MACs**: multiply-accumulates per tile for Conv2D, DepthwiseConv2D and Dense layers.

## 3. Current Memory Table
```csv
model,dtype,weights_bytes,peak_activation_bytes,tile_buffer_bytes,total_bytes,macs,recall,ece,basis
Optical,FLOAT32,61680,393216,65536,520432,4128832,0.8947,0.0384,trained_model_file
Optical,INT8,15696,327680,65536,408912,4128832,0.8553,0.0569,trained_model_file
Radar,FLOAT32,61104,262144,32768,356016,5308480,0.881,0.0362,trained_model_file
Radar,INT8,15552,65536,32768,113856,5308480,1.0,0.1066,trained_model_file

```

## 4. Omitted Memory Costs & Limitations
- **Model-only estimate**: an isolated neural network estimate, not a full flight-software measurement.
- **Not counted**:
  - TFLite Micro tensor-arena alignment overhead (typically 16-byte alignment).
  - Operator scratch buffers (for example im2col or CMSIS-NN convolution scratch).
  - Telemetry framing and radio DMA downlink queue buffers.
  - Tile preview cache, the carry-over queue, the 1-bit coverage bitmap and scheduler state.
- **Estimated vs measured**: values become MEASURED only after a micro-benchmark on target hardware.
