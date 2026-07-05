"""FastAPI endpoint serving the best fine-tuned Moondream2 LoRA adapter.

    POST /answer   multipart form: image=<file>, question=<str>  ->  {"answer": str, "latency_s": float}
    GET  /stats                                                    ->  {"count": int, "median_latency_s": float, "p95_latency_s": float}

Usage:
    python scripts/serve.py
    # then:
    curl -F "image=@some.jpg" -F "question=What color is the car?" http://localhost:8000/answer
"""
import io
import logging
import statistics
import sys
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MoondreamLoRA,
    find_text_module,
    generate_with_lora,
    load_config,
    load_moondream,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve")

app = FastAPI(title="vlm-lora-study serving")
STATE = {"latencies": []}


@app.on_event("startup")
def load_model():
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(str(repo_root / "config.yaml"))
    adapter_dir = repo_root / config["serve"]["adapter_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_moondream(config)
    model.eval()
    text_mod = find_text_module()

    import json

    with open(adapter_dir / "train_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

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

    STATE.update(
        config=config,
        model=model,
        tokenizer=tokenizer,
        text_mod=text_mod,
        lora_dict=lora.as_dict(),
        device=device,
    )
    logger.info(f"Loaded adapter {adapter_dir.name} on {device}")


@app.post("/answer")
async def answer(image: UploadFile = File(...), question: str = Form(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    start = time.time()
    pred = generate_with_lora(
        STATE["model"],
        STATE["text_mod"],
        STATE["tokenizer"],
        STATE["lora_dict"],
        pil_image,
        question,
        STATE["device"],
        max_new_tokens=STATE["config"]["training"]["max_answer_tokens"],
    )
    latency_s = time.time() - start
    STATE["latencies"].append(latency_s)
    logger.info(f"question={question!r} answer={pred!r} latency_s={latency_s:.3f}")

    return {"answer": pred, "latency_s": round(latency_s, 3)}


@app.get("/stats")
def stats():
    latencies = STATE["latencies"]
    if not latencies:
        return {"count": 0, "median_latency_s": None, "p95_latency_s": None}
    sorted_lat = sorted(latencies)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))
    return {
        "count": len(latencies),
        "median_latency_s": round(statistics.median(sorted_lat), 3),
        "p95_latency_s": round(sorted_lat[p95_idx], 3),
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(str(repo_root / "config.yaml"))
    uvicorn.run(app, host=config["serve"]["host"], port=config["serve"]["port"])
