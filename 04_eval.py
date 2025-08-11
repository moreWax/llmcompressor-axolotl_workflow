#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path

AXOLOTL_ROOT = Path("axolotl_run_sp")
checkpoints = sorted((AXOLOTL_ROOT / "run").glob("checkpoint-*"))
if not checkpoints:
    sys.exit("No checkpoints found.")
best_ckpt = checkpoints[-1]
subprocess.run([
    "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={best_ckpt},dtype=float16",
    "--tasks", "hellaswag",
    "--device", "cuda",
    "--batch_size", "8",
    "--limit", "500"
], check=True)
print("🟢  Best checkpoint evaluated:", best_ckpt)
