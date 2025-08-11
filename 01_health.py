#!/usr/bin/env python3
import sys, importlib, torch, psutil
def ok(msg): print("✅", msg); return True
def fail(msg): print("❌", msg); return False
checks = [
    all(importlib.import_module(lib) is not None for lib in
        ["llmcompressor","axolotl","lm_eval","transformers","torch","ring_flash_attn"]),
    torch.cuda.device_count() >= 4 or fail("Need 4 CUDA GPUs"),
    psutil.disk_usage(".").free / 1024**3 >= 90 or fail("Need ≥ 90 GB free")
]
sys.exit(0 if all(checks) else 1)
