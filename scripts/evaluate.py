"""Evaluate a trained LoRA adapter's accuracy on the held-out test set and
append a row to results/metrics.csv (config, accuracy, delta vs baseline,
train time, peak memory, trainable %).

Usage:
    python scripts/evaluate.py --adapter models/rank16_cosine_filtered
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MoondreamLoRA,
    append_metrics_row,
    exact_match,
    find_text_module,
    generate_with_lora,
    load_config,
    load_moondream,
    peak_mem_gb,
    read_jsonl,
)


def get_baseline_accuracy(metrics_csv: Path) -> float:
    if not metrics_csv.exists():
        raise FileNotFoundError(
            f"{metrics_csv} not found - run scripts/baseline.py before evaluate.py "
            "so a delta-vs-baseline can be computed."
        )
    df = pd.read_csv(metrics_csv)
    rows = df[df["config_name"] == "zero_shot_baseline"]
    if rows.empty:
        raise ValueError(f"No zero_shot_baseline row in {metrics_csv} - run scripts/baseline.py first.")
    return float(rows.iloc[-1]["accuracy"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    parser.add_argument("--adapter", required=True, help="path to a directory produced by train.py")
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(args.config).resolve().parent
    adapter_dir = repo_root / args.adapter if not Path(args.adapter).is_absolute() else Path(args.adapter)

    with open(adapter_dir / "train_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_records = read_jsonl(repo_root / config["data"]["test"])
    print(f"Evaluating adapter {adapter_dir.name} on {len(test_records)} examples...")

    model, tokenizer = load_moondream(config)
    model.eval()
    text_mod = find_text_module()

    model_dtype = next(model.parameters()).dtype
    lora = MoondreamLoRA.load(
        adapter_dir,
        n_layers=config["lora"]["n_layers"],
        dim=config["lora"]["dim"],
        ff_dim=config["lora"]["ff_dim"],
        rank=stats["rank"],
        alpha=stats["alpha"],
        dtype=model_dtype,
        device=device,
    )
    lora.eval()
    lora_dict = lora.as_dict()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    correct = 0
    start = time.time()
    for rec in tqdm(test_records):
        image = Image.open(rec["image"]).convert("RGB")
        pred = generate_with_lora(
            model, text_mod, tokenizer, lora_dict, image, rec["question"], device,
            max_new_tokens=config["training"]["max_answer_tokens"],
        )
        if exact_match(pred, rec["answer"]):
            correct += 1
    eval_time_s = time.time() - start

    accuracy = correct / len(test_records)
    baseline_accuracy = get_baseline_accuracy(repo_root / config["paths"]["metrics_csv"])
    delta = accuracy - baseline_accuracy

    print(f"\naccuracy: {accuracy:.4f} ({correct}/{len(test_records)})")
    print(f"delta vs zero-shot baseline ({baseline_accuracy:.4f}): {delta:+.4f}")

    append_metrics_row(
        repo_root / config["paths"]["metrics_csv"],
        {
            "config_name": adapter_dir.name,
            "rank": stats["rank"],
            "lr_schedule": stats["lr_schedule"],
            "data_variant": stats["data_variant"],
            "accuracy": round(accuracy, 4),
            "delta_vs_baseline": round(delta, 4),
            "trainable_pct": stats["trainable_pct"],
            "train_time_s": stats["train_time_s"],
            "eval_time_s": round(eval_time_s, 1),
            "peak_mem_gb": max(stats["peak_mem_gb"], round(peak_mem_gb(), 2)),
        },
    )


if __name__ == "__main__":
    main()
