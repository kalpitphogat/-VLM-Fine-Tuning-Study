"""LoRA fine-tune of Moondream2 using its native lora-dict mechanism.

IMPORTANT - read this before running the full matrix:
Everything in `compute_lm_logits()` below was reconstructed by reading
Moondream2's published source (huggingface.co/vikhyatk/moondream2) without
being able to execute it locally. Run `python scripts/inspect_model.py` on
your GPU box first, and ALWAYS run `--smoke_test` before a real job - it
does one forward+backward on a single example and checks gradients actually
reach the LoRA parameters, so a wrong assumption fails in 30 seconds instead
of burning your Kaggle GPU quota on a run that silently trains nothing.

Usage:
    python scripts/train.py --smoke_test
    python scripts/train.py --rank 16 --lr 2e-4 --schedule cosine --data filtered --out models/rank16_cosine_filtered
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MoondreamLoRA,
    build_train_embeds,
    compute_lm_logits,
    find_text_module,
    load_config,
    load_moondream,
    peak_mem_gb,
    read_jsonl,
    report_trainable_params,
    set_seed,
)


def training_step(model, text_mod, tokenizer, lora, example, device):
    image = Image.open(example["image"]).convert("RGB")
    full_embeds, labels, seq_len, bidir_len = build_train_embeds(
        model, text_mod, tokenizer, image, example["question"], example["answer"], device
    )
    lora_dict = lora.as_dict()
    logits = compute_lm_logits(model, text_mod, lora_dict, full_embeds, bidir_len, device)

    # next-token prediction: logits[t] predicts labels[t+1]
    shift_logits = logits[0, :-1, :]
    shift_labels = labels[1:]
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return loss


def make_scheduler(optimizer, schedule, num_warmup_steps, num_training_steps):
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

    if schedule == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif schedule == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    raise ValueError(f"Unknown schedule: {schedule}")


def run_smoke_test(model, text_mod, tokenizer, config, device):
    print("Running smoke test: one forward+backward pass on a single example...")
    repo_root = Path(__file__).resolve().parent.parent
    test_records = read_jsonl(repo_root / config["data"]["test"])
    example = test_records[0]

    model_dtype = next(model.parameters()).dtype
    lora = MoondreamLoRA(
        n_layers=config["lora"]["n_layers"],
        dim=config["lora"]["dim"],
        ff_dim=config["lora"]["ff_dim"],
        rank=config["lora"]["rank"],
        alpha=config["lora"]["alpha"],
        dtype=model_dtype,
    ).to(device)

    loss = training_step(model, text_mod, tokenizer, lora, example, device)
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in lora.parameters() if p.grad is not None]
    n_params = sum(1 for _ in lora.parameters())
    print(f"loss = {loss.item():.4f}")
    print(f"{len(grad_norms)}/{n_params} LoRA parameters received a gradient")
    if len(grad_norms) == 0:
        raise RuntimeError(
            "No gradients reached the LoRA parameters. compute_lm_logits()'s "
            "assumptions about text_decoder/lm_head are wrong - re-run "
            "scripts/inspect_model.py and fix build_example_embeds/compute_lm_logits."
        )
    if len(grad_norms) < n_params:
        print(
            f"WARNING: only {len(grad_norms)}/{n_params} params got gradients - "
            "some layers may not be wired into the lora dict correctly."
        )
    print("Smoke test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--schedule", choices=["cosine", "constant"], default=None)
    parser.add_argument("--data", choices=["filtered", "unfiltered"], default="filtered")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", default=None, help="output dir for the trained adapter")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(args.config).resolve().parent
    tcfg = config["training"]
    lcfg = config["lora"]

    rank = args.rank or lcfg["rank"]
    alpha = args.alpha or lcfg["alpha"]
    lr = args.lr or tcfg["lr"]
    schedule = args.schedule or tcfg["lr_schedule"]
    epochs = args.epochs or tcfg["epochs"]

    set_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Moondream2 (frozen base, device={device})...")
    model, tokenizer = load_moondream(config)
    model.eval()
    text_mod = find_text_module()

    if args.smoke_test:
        run_smoke_test(model, text_mod, tokenizer, config, device)
        return

    if args.out is None:
        args.out = f"models/rank{rank}_{schedule}_{args.data}"
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = repo_root / (
        config["data"]["train_filtered"] if args.data == "filtered" else config["data"]["train_unfiltered"]
    )
    train_records = read_jsonl(data_path)
    print(f"Training on {len(train_records)} examples from {data_path.name}")
    print(f"rank={rank} alpha={alpha} lr={lr} schedule={schedule} epochs={epochs} data={args.data}")

    model_dtype = next(model.parameters()).dtype
    lora = MoondreamLoRA(
        n_layers=lcfg["n_layers"], dim=lcfg["dim"], ff_dim=lcfg["ff_dim"], rank=rank, alpha=alpha, dtype=model_dtype
    ).to(device)
    trainable_pct = report_trainable_params(model, lora)

    import bitsandbytes as bnb

    if tcfg["optimizer"] == "paged_adamw_8bit":
        optimizer = bnb.optim.PagedAdamW8bit(lora.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(lora.parameters(), lr=lr)

    grad_accum = tcfg["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(train_records) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * tcfg["warmup_ratio"]))
    scheduler = make_scheduler(optimizer, schedule, warmup_steps, total_steps)

    use_wandb = config["wandb"]["enabled"]
    if use_wandb:
        import wandb

        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            name=out_dir.name,
            config={"rank": rank, "alpha": alpha, "lr": lr, "schedule": schedule, "data": args.data},
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pbar = tqdm(train_records, desc=f"epoch {epoch + 1}/{epochs}")
        for i, example in enumerate(pbar):
            loss = training_step(model, text_mod, tokenizer, lora, example, device)
            (loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % tcfg["logging_steps"] == 0:
                    pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
                    if use_wandb:
                        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "step": global_step})

    train_time_s = time.time() - start_time
    peak_mem = peak_mem_gb()

    lora.save(out_dir)
    stats = {
        "rank": rank,
        "alpha": alpha,
        "lr": lr,
        "lr_schedule": schedule,
        "data_variant": args.data,
        "epochs": epochs,
        "trainable_pct": round(trainable_pct, 4),
        "train_time_s": round(train_time_s, 1),
        "peak_mem_gb": round(peak_mem, 2),
    }
    with open(out_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved adapter + train_stats.json to {out_dir}")
    print(json.dumps(stats, indent=2))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
