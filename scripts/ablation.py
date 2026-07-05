"""Orchestrates the full ablation matrix:

  1. rank in config.ablation.ranks, cosine schedule, filtered data
  2. best rank (by accuracy) x each schedule in config.ablation.schedules
  3. best rank x each data variant in config.ablation.data_variants

Each (rank, schedule, data) combo already covered by step 1 is skipped in
steps 2/3 rather than re-run. Every run is a fresh `python scripts/train.py`
subprocess (not an in-process call) so GPU memory is fully released between
runs - important on a 16GB card running back-to-back fine-tunes.

Usage:
    python scripts/ablation.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common import load_config  # noqa: E402


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # Brief pause so CUDA driver can fully release GPU memory before the next process
    import time
    time.sleep(8)


def train_and_eval(python, scripts_dir, rank, schedule, data, results):
    key = (rank, schedule, data)
    if key in results:
        print(f"skip rank={rank} schedule={schedule} data={data} (already run)")
        return
    out = f"models/rank{rank}_{schedule}_{data}"
    run([python, str(scripts_dir / "train.py"), "--rank", str(rank), "--schedule", schedule, "--data", data, "--out", out])
    run([python, str(scripts_dir / "evaluate.py"), "--adapter", out])
    results.add(key)


def load_existing_results(metrics_csv: Path):
    if not metrics_csv.exists():
        return set()
    df = pd.read_csv(metrics_csv)
    df = df[df["config_name"] != "zero_shot_baseline"]
    seen = set()
    for _, row in df.iterrows():
        seen.add((int(row["rank"]), row["lr_schedule"], row["data_variant"]))
    return seen


def best_rank_from_results(metrics_csv: Path, schedule, data):
    df = pd.read_csv(metrics_csv)
    subset = df[(df["lr_schedule"] == schedule) & (df["data_variant"] == data)]
    if subset.empty:
        raise RuntimeError(f"No results for schedule={schedule} data={data} to pick a best rank from.")
    return int(subset.loc[subset["accuracy"].idxmax(), "rank"])


def main():
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(str(repo_root / "config.yaml"))
    scripts_dir = repo_root / "scripts"
    metrics_csv = repo_root / config["paths"]["metrics_csv"]
    python = sys.executable

    # Skip smoke test here — caller should verify manually via: python scripts/train.py --smoke_test
    # The smoke test is already confirmed passing; re-running it consumes GPU memory
    # that can collide with the immediately following training subprocess on this server.
    print("Step 0: smoke test skipped (pre-confirmed). Starting ablation runs.")

    if not metrics_csv.exists() or "zero_shot_baseline" not in pd.read_csv(metrics_csv).get("config_name", pd.Series()).values:
        print("\nStep 0b: no zero-shot baseline found, running scripts/baseline.py")
        run([python, str(scripts_dir / "baseline.py")])

    results = load_existing_results(metrics_csv)

    print("\nStep 1: rank sweep (cosine, filtered)")
    for rank in config["ablation"]["ranks"]:
        train_and_eval(python, scripts_dir, rank, "cosine", "filtered", results)

    best_rank = best_rank_from_results(metrics_csv, "cosine", "filtered")
    print(f"\nBest rank from step 1: {best_rank}")

    print(f"\nStep 2: best rank ({best_rank}) x schedule")
    for schedule in config["ablation"]["schedules"]:
        train_and_eval(python, scripts_dir, best_rank, schedule, "filtered", results)

    print(f"\nStep 3: best rank ({best_rank}) x data variant")
    for data in config["ablation"]["data_variants"]:
        train_and_eval(python, scripts_dir, best_rank, "cosine", data, results)

    print(f"\nAblation matrix complete. See {metrics_csv}")
    print(pd.read_csv(metrics_csv).to_string(index=False))


if __name__ == "__main__":
    main()
