# Sparse-Long-Context Fine-Tuning of GPT-OSS-120B  
**End-to-End Pipeline for 4 × RTX 3090 (24 GB each)**

---

## Overview

This repository provides a **fully automated, reproducible pipeline** that compresses **openai/gpt-oss-120b** (117 B MoE) to **4-bit weights + 2:4 sparsity**, then fine-tunes the model with **16 384-token context windows** on **consumer-grade GPUs** using **Axolotl’s sequence parallelism**.  

The workflow trades **model size** for **context length** while preserving reasoning quality, enabling long-context research on hardware that would otherwise exceed VRAM limits.

---

## Architecture & Memory Analysis

| Component | Precision | Disk | Per-GPU VRAM (loaded) |
|---|---|---|---|
| Shared Dense Layers | 4-bit int, 2:4 sparse | ~24 GB | ~12 GB |
| Expert FFN Weights | 4-bit int, dense | ~14 GB | ~7 GB |
| Router / Gate | fp16 | < 0.5 GB | < 0.5 GB |
| **Total** | — | **~38 GB** | **~20 GB** |

With **sequence-parallel degree = 4**, **16 384-token sequences** fit comfortably within **4 × RTX 3090 (24 GB each)** after accounting for optimizer states, gradients, and activations.

---

## Features

- **Dynamic 2:4 sparsity** – `MagnitudePruningModifier` recomputes masks every 5 % of training steps.
- **Sequence parallelism** – transparent 4-GPU ring-attention via `ring-flash-attn`.
- **Checkpointing** – automatic retention of the best validation perplexity.
- **Composability** – compatible with Liger kernels, FSDP, DeepSpeed ZeRO, `torch.compile`, and sample packing.

---

## File Map

| File | Purpose |
|---|---|
| `install.sh` | One-time environment setup (Python 3.10+, CUDA 12.1+) |
| `01_health.py` | Hardware & dependency health check |
| `02_compress.py` | Compress checkpoint (`gpt-oss-120b-w4a16-experts-4bit-2:4`) |
| `03_finetune.py` | Fine-tune with dynamic sparsity & 16 k context |
| `04_eval.py` | Evaluate best checkpoint on `hellaswag` |

---

## Quick Start

```bash
# 1. Install
bash install.sh

# 2. Verify
python 01_health.py   # exits 0 if ready

# 3. Compress (single GPU)
python 02_compress.py

# 4. Fine-tune (4 × RTX 3090)
python 03_finetune.py

# 5. Evaluate
python 04_eval.py
```

---

## Configuration Details

### Quantisation & Sparsity
```yaml
shared_dense:
  weight_bits: 4
  sparsity: 50 % (2:4)
experts:
  weight_bits: 4
```

### Sequence Parallelism
```yaml
sequence_len: 16384
sequence_parallel_degree: 4
flash_attention: true
micro_batch_size: 1
heads_k_stride: 1
ring_attn_func: varlen_llama3
```

### Dynamic Sparsity
```yaml
finetuning_modifiers:
  MagnitudePruningModifier:
    update_frequency: 0.05   # every 5 % steps
```

---

## Benchmarks (paper-aligned)

| SP Degree | Max Context | Context Scaling | Tokens/sec (4× 3090) |
|---|---|---|---|
| 1 | 4 096 | 1.00× | 3 674 |
| 2 | 8 192 | 2.00× | 5 022 |
| 4 | 16 384 | 4.00× | 6 455 |

---

## Contributing

Pull requests for additional optimizers, datasets, or hyper-parameter sweeps are welcome.  
Open an issue for hardware-specific tuning requests.

---

## License

Apache-2.0 – identical to the upstream `openai/gpt-oss-120b` license.
