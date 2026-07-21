# ROCmFP4 Integration with Upstream Master

## Overview

This document describes the integration of ROCmFP4 FP16 activation support into the upstream `llama.cpp` master branch.

## Branch Information

- **Branch**: `rocmfp4-strix-upstream-merged`
- **Base**: `upstream/master` (commit `5735e10c4`)
- **ROCmFP4 Source**: `origin/rocmfp4-strix-original`

## What Was Merged

### 3 New Commits

1. **feat: add ROCmFP4 FP16 activation support** (`35ba304f7`)
   - Added `ggml/rocmfp4/` directory with CPU quantize backend
   - Added FP16 activation quantization types:
     - `Q4_0_ROCMFP4_FP16` (ftype 107)
     - `Q4_0_ROCMFP4_FAST_FP16` (ftype 108)
   - Updated type registration across `ggml.h`, `llama.h`, `ggml.c`, `ggml-cpu.c`, `quantize.cpp`
   - Added `docs/FP4_FP6_FP8_Analysis.md` documentation

2. **feat: add ROCmFP4 CUDA template instances** (`906f75aa3`)
   - Added MMQ template instances for Q4_0_ROCMFP4 and Q4_0_ROCMFP4_FAST
   - Template files: `mmq-instance-q4_0_rocmfp4.cu`, `mmq-instance-q4_0_rocmfp4_fast.cu`

3. **build: add ROCmFP4 CPU backend to CMakeLists** (`0c4b304c1`)
   - Added `ggml/rocmfp4/rocmfp4.c` to ggml-base target
   - Added `ggml_cuda_type_traits` specializations for ROCmFP4 types
   - Fixed includes in `vecdotq.cuh` and `common.cuh`

### Files Modified

- `ggml/include/ggml.h` - Type definitions
- `ggml/src/ggml.c` - Type registration
- `ggml/src/ggml-common.h` - ROCmFP4 type constants
- `ggml/src/ggml-cpu/ggml-cpu.c` - CPU backend
- `ggml/src/ggml-cuda/common.cuh` - CUDA type traits
- `ggml/src/ggml-cuda/vecdotq.cuh` - ROCmFP4 support
- `ggml/src/ggml-cuda/CMakeLists.txt` - CUDA build config
- `ggml/src/ggml-hip/CMakeLists.txt` - HIP build config
- `ggml/src/ggml-quants.c` - Quantization functions
- `include/llama.h` - API definitions
- `src/llama-quant.cpp` - Quantization API
- `tools/quantize/quantize.cpp` - CLI tool
- `gguf-py/gguf/constants.py` - Python bindings
- `ggml/CMakeLists.txt` - Base build config

### Files Added

- `ggml/rocmfp4/rocmfp4.c` - CPU quantization backend
- `ggml/rocmfp4/rocmfp4.h` - ROCmFP4 type definitions
- `ggml/rocmfp4/rocmfp4_hip.cu` - HIP GPU kernels
- `ggml/rocmfp4/rocmfp4_hip_codebook.cuh` - Codebook utilities
- `ggml/rocmfp4/rocmfp4_hip_scale.cuh` - Scale utilities
- `docs/FP4_FP6_FP8_Analysis.md` - FP4/FP6/FP8 analysis
- `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0_rocmfp4.cu`
- `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0_rocmfp4_fast.cu`

## Implementation Details

### FP16 Activation Path

ROCmFP4 with FP16 activations **falls through to the standard MMQ path** using:
- DP4A instructions via MFMA on AMD Strix Halo (gfx1151)
- No dedicated FP16 kernel (removed `mul_mat_rocmfp4_fp16.cu/.cuh`)

### Known Limitations

- **Flash Attention**: ROCmFP4 Flash Attention template instances removed (no vec_dot implementation available)
- **FP16 activations**: Use standard MMQ path instead of dedicated kernel

## Build Instructions

```bash
# Build with ROCmFP4 support
bash build_llama.cpp.sh

# Test ROCmFP4 with FP16 activations
LLAMA_CPP_DIR=$(pwd)/llama.cpp/build bash llama.sh completion \
  --model Qwen3.5-4B-ROCMFP4.gguf -p "Hi" -n 500

# Quantize BF16 to ROCmFP4 FP16 variants
./llama.cpp/build/bin/llama-quantize Qwen3.5-4B-BF16.gguf out.gguf Q4_0_ROCMFP4_FP16
./llama.cpp/build/bin/llama-quantize Qwen3.5-4B-BF16.gguf out.gguf Q4_0_ROCMFP4_FAST_FP16
```

## Build Status

✅ **Build successful** - All targets built:
- `llama-cli`
- `llama-server`
- `llama-completion`
- `llama-quantize`
- `llama-bench`

## Comparison with rocmfp4-llama Submodule

See `rocmfp4-llama/` submodule (branch `nemotron-mtp-rocmfp4-strix`) for:
- MTP (Multi-Token Prediction) support
- Additional ROCmFP4 optimizations
- Custom kernel implementations

## Remote Configuration

```bash
# Add upstream remote if not present
git remote add upstream https://github.com/ggml-org/llama.cpp.git

# Push this branch
git push origin rocmfp4-strix-upstream-merged
```

## Maintenance

To keep this branch up-to-date:

```bash
# Fetch latest upstream
git fetch upstream

# Rebase or merge as needed
git checkout rocmfp4-strix-upstream-merged
git merge upstream/master
# or
git rebase upstream/master
```

## Author

Based on ROCmFP4 implementation by Tim Aerts (aardbeiplantje)

## Date

Created: July 2025
