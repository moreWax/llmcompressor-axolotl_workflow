#!/usr/bin/env python3
"""
Compress gpt-oss-120b → 4-bit + 2:4 + experts 4-bit (W4A16)
"""
from pathlib import Path
from transformers import AutoTokenizer
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM

MODEL_ID       = "openai/gpt-oss-120b"
COMPRESSED_DIR = Path("gpt-oss-120b-w4a16-experts-4bit-2:4")
if COMPRESSED_DIR.exists():
    print("Exists:", COMPRESSED_DIR); exit(0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto",
                                             device_map="auto", trust_remote_code=True)

recipe = """
quant_stage:
  quant_modifiers:
    - GPTQModifier:
        config_groups:
          shared:
            weights:
              num_bits: 4
              type: int
              symmetric: true
              strategy: channel
              group_size: 64
            targets: [Linear]
            ignore: ["lm_head", "*router*", "experts.*"]
    - SparseGPTModifier:
        sparsity: 0.5
        mask_structure: "2:4"
        targets: [Linear]
        ignore: ["lm_head", "*router*", "experts.*"]
    - GPTQModifier:
        config_groups:
          experts:
            weights:
              num_bits: 4
              type: int
              symmetric: true
              strategy: channel
              group_size: 64
            targets: ["re:experts.*\\.w[12]"]
"""
oneshot(model=model, recipe=recipe, tokenizer=tokenizer,
        save_path=str(COMPRESSED_DIR), max_seq_length=2048, num_calibration_samples=512)
tokenizer.save_pretrained(COMPRESSED_DIR)
print("Done →", COMPRESSED_DIR)
