"""Zero-shot Moondream2 accuracy on the held-out test set (no LoRA adapter).

This is the number every fine-tuned config's accuracy is compared against
(results/metrics.csv row "zero_shot_baseline").

Usage:
    python scripts/baseline.py
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    append_metrics_row,
    exact_match,
    load_config,
    load_moondream,
    peak_mem_gb,
    read_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    repo_root = Path(args.config).resolve().parent

    test_records = read_jsonl(repo_root / config["data"]["test"])
    print(f"Evaluating zero-shot baseline on {len(test_records)} examples...")

    model, tokenizer = load_moondream(config)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    correct = 0
    start = time.time()
    with torch.inference_mode():
        for rec in tqdm(test_records):
            image = Image.open(rec["image"]).convert("RGB")
            pred = model.query(image, rec["question"])["answer"]
            if exact_match(pred, rec["answer"]):
                correct += 1
    elapsed = time.time() - start

    accuracy = correct / len(test_records)
    print(f"\nZero-shot baseline accuracy: {accuracy:.4f} ({correct}/{len(test_records)})")

    append_metrics_row(
        repo_root / config["paths"]["metrics_csv"],
        {
            "config_name": "zero_shot_baseline",
            "rank": "",
            "lr_schedule": "",
            "data_variant": "",
            "accuracy": round(accuracy, 4),
            "delta_vs_baseline": 0.0,
            "trainable_pct": 0.0,
            "train_time_s": 0.0,
            "eval_time_s": round(elapsed, 1),
            "peak_mem_gb": round(peak_mem_gb(), 2),
        },
    )


if __name__ == "__main__":
    main()
