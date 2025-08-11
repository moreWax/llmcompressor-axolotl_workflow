#!/usr/bin/env python3
"""
Fine-tune on 4× RTX 3090 with sequence parallelism (16 K context)
"""
import yaml, subprocess, json
from pathlib import Path

COMPRESSED_DIR = Path("gpt-oss-120b-w4a16-experts-4bit-2:4")
AXOLOTL_ROOT   = Path("axolotl_run_sp")
DATASET_URL    = "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
yaml_path = AXOLOTL_ROOT / "sparse_finetune_4x3090.yaml"

dataset_path = AXOLOTL_ROOT / "alpaca_mini.jsonl"
if not dataset_path.exists():
    import pandas as pd
    AXOLOTL_ROOT.mkdir(exist_ok=True)
    df = pd.read_parquet(DATASET_URL).sample(2000, random_state=42)
    with dataset_path.open("w") as f:
        for _, row in df.iterrows():
            json.dump({"instruction": row["instruction"], "input": row["input"], "output": row["output"]}, f)
            f.write("\n")

cfg = {
    "base_model": str(COMPRESSED_DIR),
    "model_type": "AutoModelForCausalLM",
    "tokenizer_type": "AutoTokenizer",
    "datasets": [{"path": str(dataset_path), "type": "alpaca"}],
    "max_steps": 400,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 64,
    "learning_rate": 2e-5,
    "lr_scheduler": "cosine",
    "warmup_steps": 50,
    "bf16": True,
    "output_dir": str(AXOLOTL_ROOT / "run"),
    "save_steps": 100,
    "eval_steps": 100,
    "sequence_len": 16384,
    "flash_attention": True,
    "sequence_parallel_degree": 4,
    "heads_k_stride": 1,
    "ring_attn_func": "varlen_llama3",
    "plugins": ["axolotl.integrations.llm_compressor.LLMCompressorPlugin"],
    "llmcompressor": {
        "recipe": {
            "finetuning_stage": {
                "finetuning_modifiers": [
                    {
                        "MagnitudePruningModifier": {
                            "init_sparsity": 0.50,
                            "final_sparsity": 0.50,
                            "start": 0,
                            "end": -1,
                            "update_frequency": 0.05,
                            "targets": [
                                "re:.*q_proj.weight", "re:.*k_proj.weight",
                                "re:.*v_proj.weight", "re:.*o_proj.weight",
                                "re:.*gate_proj.weight", "re:.*up_proj.weight",
                                "re:.*down_proj.weight"
                            ]
                        }
                    }
                ]
            }
        },
        "save_compressed": True
    }
}
yaml_path.write_text(yaml.dump(cfg))

subprocess.run([
    "accelerate", "launch",
    "--num_processes", "4",
    "--config_file", "accelerate_4x3090.yaml",
    "-m", "axolotl.cli.train", str(yaml_path)
])
