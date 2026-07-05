"""Build {image, question, answer} jsonl splits from a VQAv2 slice.

Produces three files (paths from config.yaml data.*):
  - test.jsonl              held out, used by every eval run
  - train_unfiltered.jsonl  all remaining examples
  - train_filtered.jsonl    train_unfiltered minus trivial yes/no/single-char answers

Usage:
    python scripts/prepare_data.py
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from common import load_config, set_seed, write_jsonl  # noqa: E402


def is_trivial(answer: str, min_len: int) -> bool:
    a = answer.strip().lower()
    if a in ("yes", "no"):
        return True
    if len(a) <= min_len:
        return True
    if a.isdigit():
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    dcfg = config["data"]
    set_seed(config["training"]["seed"])

    from datasets import load_dataset
    from tqdm import tqdm

    n_train = dcfg["n_samples"]
    n_test = dcfg["test_size"]
    total_needed = n_train + n_test

    repo_root = Path(args.config).resolve().parent
    images_dir = repo_root / dcfg["raw_dir"] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {dcfg['dataset_id']} split={dcfg['dataset_split']} ...")
    ds = load_dataset(
        dcfg["dataset_id"], split=dcfg["dataset_split"], streaming=True, trust_remote_code=True
    )
    ds = ds.shuffle(seed=config["training"]["seed"], buffer_size=10_000)

    records = []
    seen_ids = set()
    pbar = tqdm(total=total_needed, desc="collecting")
    for example in ds:
        if len(records) >= total_needed:
            break
        qid = example.get("question_id", example.get("question_type", len(records)))
        if qid in seen_ids:
            continue
        seen_ids.add(qid)

        answer = (example.get("multiple_choice_answer") or "").strip()
        question = (example.get("question") or "").strip()
        image = example.get("image")
        if not answer or not question or image is None:
            continue

        image_path = images_dir / f"{qid}.jpg"
        image.convert("RGB").save(image_path, format="JPEG", quality=90)

        records.append(
            {
                "question_id": qid,
                "image": str(image_path.resolve().as_posix()),
                "question": question,
                "answer": answer,
            }
        )
        pbar.update(1)
    pbar.close()

    if len(records) < total_needed:
        raise RuntimeError(
            f"Only collected {len(records)} examples, needed {total_needed}. "
            "The stream may have run out, or too many duplicate/empty rows were skipped. "
            "Try increasing n_samples+test_size headroom or the shuffle buffer_size."
        )

    test_records = records[:n_test]
    train_pool = records[n_test:]
    filtered = [r for r in train_pool if not is_trivial(r["answer"], dcfg["min_trivial_answer_len"])]

    write_jsonl(repo_root / dcfg["test"], test_records)
    write_jsonl(repo_root / dcfg["train_unfiltered"], train_pool)
    write_jsonl(repo_root / dcfg["train_filtered"], filtered)

    dropped = len(train_pool) - len(filtered)
    print(f"\ntest set:          {len(test_records):5d} examples -> {dcfg['test']}")
    print(f"train_unfiltered:  {len(train_pool):5d} examples -> {dcfg['train_unfiltered']}")
    print(
        f"train_filtered:    {len(filtered):5d} examples -> {dcfg['train_filtered']} "
        f"({dropped} trivial answers dropped, {100 * dropped / len(train_pool):.1f}%)"
    )


if __name__ == "__main__":
    main()
