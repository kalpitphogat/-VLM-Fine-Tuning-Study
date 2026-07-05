"""Shared utilities used by every script in this repo.

Nothing in here is Moondream-training-loop-specific except `MoondreamLoRA`
and `load_moondream` - see the big comment on `MoondreamLoRA` for why LoRA is
hand-rolled instead of using PEFT's `get_peft_model`.
"""
import csv
import json
import os
import random
import string
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    return text.lower().strip().translate(_PUNCT_TABLE).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_metrics_row(csv_path, row: dict):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start


def peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def load_moondream(config, dtype=None):
    """Load the base Moondream2 model + tokenizer.

    Loads in fp16 by default (see config.yaml `quantization` comment for why
    bnb 4-bit is opt-in and unverified for this architecture). Returns the
    raw HF-wrapped model (`HfMoondream`) - LoRA is attached separately via
    `MoondreamLoRA`, not via `get_peft_model`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = config["model"]
    qcfg = config["quantization"]
    dtype = dtype or getattr(torch, qcfg["torch_dtype"])

    kwargs = dict(
        trust_remote_code=mcfg["trust_remote_code"],
        revision=mcfg.get("revision"),
        torch_dtype=dtype,
    )

    if qcfg["load_in_4bit"]:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
        )

    model = AutoModelForCausalLM.from_pretrained(mcfg["id"], **kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(mcfg["id"], revision=mcfg.get("revision"))

    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


class MoondreamLoRA(nn.Module):
    """Hand-rolled LoRA adapter matching Moondream2's NATIVE lora-dict hook.

    Moondream2's text.py defines `attn(...)` and `text_decoder(...)` with a
    `lora: Optional[dict]` argument, applied internally roughly as:

        if lora is not None:
            out += F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])

    keyed as lora["text"]["blocks"][str(layer_idx)]["attn"|"mlp"]["qkv"|"proj"|"fc1"|"fc2"].
    This is NOT the standard PEFT nn.Linear-wrapping mechanism - Moondream2's
    core transformer does not use nn.Linear submodules for attn/mlp (they are
    plain tensors in dataclasses, applied via F.linear in functional code), so
    `peft.get_peft_model()` has nothing to attach to. This class builds
    exactly the parameter shapes Moondream2's own lora dict expects, registers
    them as real nn.Parameters (so they see gradients/optimizer updates), and
    exposes `.as_dict()` to pass straight into text_decoder(..., lora=...).

    Shapes follow TextConfig: A is (rank, in_features), B is (out_features, rank),
    matching `F.linear(x, A)` -> (..., rank) then `F.linear(_, B)` -> (..., out_features).
    """

    _PROJ_SHAPES = {
        # name -> (in_features, out_features), derived from TextConfig (dim, ff_dim)
        # qkv is fused q+k+v with equal head/kv-head count -> out = 3 * dim
        "qkv": ("dim", "qkv_out"),
        "proj": ("dim", "dim"),
        "fc1": ("dim", "ff_dim"),
        "fc2": ("ff_dim", "dim"),
    }

    def __init__(self, n_layers: int, dim: int, ff_dim: int, rank: int, alpha: int, dtype=torch.float16):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        self.scale = alpha / rank
        sizes = {"dim": dim, "ff_dim": ff_dim, "qkv_out": 3 * dim}

        params = {}
        for layer in range(n_layers):
            for block_name in ("attn", "mlp"):
                proj_names = ("qkv", "proj") if block_name == "attn" else ("fc1", "fc2")
                for proj in proj_names:
                    in_f = sizes[self._PROJ_SHAPES[proj][0]]
                    out_f = sizes[self._PROJ_SHAPES[proj][1]]
                    key_a = f"{layer}__{block_name}__{proj}__A"
                    key_b = f"{layer}__{block_name}__{proj}__B"
                    params[key_a] = nn.Parameter(torch.zeros(rank, in_f, dtype=dtype))
                    params[key_b] = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype))
                    # Standard LoRA init: A ~ small random (kaiming), B = 0 so the
                    # adapter starts as a no-op and training perturbs from there.
                    nn.init.kaiming_uniform_(params[key_a], a=5**0.5)

        self.params = nn.ParameterDict(params)

    def as_dict(self):
        """Rebuild the nested dict Moondream2's text_decoder(..., lora=...) expects."""
        blocks = {}
        for layer in range(self.n_layers):
            layer_dict = {}
            for block_name, proj_names in (("attn", ("qkv", "proj")), ("mlp", ("fc1", "fc2"))):
                proj_dict = {}
                for proj in proj_names:
                    a = self.params[f"{layer}__{block_name}__{proj}__A"]
                    b = self.params[f"{layer}__{block_name}__{proj}__B"] * self.scale
                    proj_dict[proj] = {"A": a, "B": b}
                layer_dict[block_name] = proj_dict
            blocks[str(layer)] = layer_dict
        return {"text": {"blocks": blocks}}

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "lora_adapter.pt")

    @classmethod
    def load(cls, path, n_layers, dim, ff_dim, rank, alpha, dtype=torch.float16, device="cuda"):
        adapter = cls(n_layers=n_layers, dim=dim, ff_dim=ff_dim, rank=rank, alpha=alpha, dtype=dtype)
        state = torch.load(Path(path) / "lora_adapter.pt", map_location=device)
        adapter.load_state_dict(state)
        return adapter.to(device)


def report_trainable_params(base_model, lora_module):
    base_total = sum(p.numel() for p in base_model.parameters())
    lora_total = lora_module.trainable_param_count()
    pct = 100 * lora_total / (base_total + lora_total)
    print(
        f"trainable params: {lora_total:,} || all params: {base_total + lora_total:,} "
        f"|| trainable%: {pct:.4f}"
    )
    return pct


# ---------------------------------------------------------------------------
# Low-level Moondream2 forward/generation path, shared by train.py, evaluate.py
# and serve.py. Moondream2's public `model.query(...)` API does NOT accept a
# custom in-memory `lora` dict (its `settings={"variant": ...}` only downloads
# one of the project's own hosted pre-quantized/pre-tuned variants via
# lora.py's `variant_state_dict`). To run inference with OUR trained adapter
# we have to call the same lower-level functions train.py uses directly.
# See scripts/inspect_model.py and the big comment in train.py before trusting
# this - it was reconstructed from reading source, not from execution.
# ---------------------------------------------------------------------------


def inner_model(model):
    return model.model if hasattr(model, "model") and hasattr(model.model, "_run_vision_encoder") else model


def find_text_module():
    import importlib
    import sys

    for candidate in ("moondream", "text"):
        try:
            mod = importlib.import_module(candidate)
            if hasattr(mod, "text_decoder"):
                return mod
        except ImportError:
            pass
    for name, mod in list(sys.modules.items()):
        if name.startswith("torch") or name.startswith("sys") or name.startswith("importlib"):
            continue
        try:
            if hasattr(mod, "text_decoder") and hasattr(mod, "lm_head"):
                return mod
        except Exception:
            pass
    raise RuntimeError(
        "Could not locate the module defining text_decoder/lm_head. "
        "Run scripts/inspect_model.py, find it in sys.modules, and hardcode "
        "the module name here."
    )


def _attn_no_cache_with_lora(x, w, freqs_cis, attn_mask, n_heads, n_kv_heads, lora, apply_rotary_emb):
    """Cache-free reimplementation of Moondream2's text.py:attn(), for a
    single-shot full-sequence forward (training/eval), with the kv_cache
    dependency removed (it always mutates persistent block.kv_cache buffers,
    which is right for incremental generation but wrong for us - see
    text.py:text_decoder). Position handling matches text.py:_attn (always
    starts at position 0), lora handling matches text.py:attn verbatim.
    Verified against live source dumped by scripts/inspect_model.py.
    """
    import torch.nn.functional as F

    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    qkv_out = w.qkv(x)
    if lora is not None:
        qkv_out = qkv_out + F.linear(F.linear(x, lora["qkv"]["A"]), lora["qkv"]["B"])

    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    position_ids = torch.arange(q_len, dtype=torch.long, device=x.device)
    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_kv_heads)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=(n_heads != n_kv_heads))
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)

    out0 = w.proj(out)
    if lora is not None:
        out1 = F.linear(F.linear(x, lora["proj"]["A"]), lora["proj"]["B"])
        out = out0 + out1
    else:
        out = out0
    return out


def _produce_hidden_with_lora(text_mod, inputs_embeds, w, config, lora, bidir_len):
    """Cache-free full-sequence forward through all text blocks, mirroring
    text.py:_produce_hidden's prefix-bidirectional + causal mask (attend
    bidirectionally among the first `bidir_len` tokens - BOS + image - then
    causally after) but with LoRA support added (_produce_hidden has none;
    text_decoder has LoRA but is cache-stateful - see module docstring)."""
    hidden = inputs_embeds
    bsz, q_len, d_model = inputs_embeds.shape
    device = inputs_embeds.device

    attn_mask = torch.zeros(q_len, q_len, dtype=torch.bool, device=device)
    attn_mask[:bidir_len, :bidir_len] = True
    for i in range(bidir_len, q_len):
        attn_mask[i, : i + 1] = True

    for i, block in enumerate(w.blocks):
        if lora is not None:
            layer_lora = lora["text"]["blocks"][str(i)]
            attn_lora = layer_lora["attn"]
            mlp_lora = layer_lora["mlp"]
        else:
            attn_lora = None
            mlp_lora = None

        l_in = text_mod.layer_norm(hidden, block.ln)
        l_attn = _attn_no_cache_with_lora(
            l_in, block.attn, w.freqs_cis, attn_mask, config.n_heads, config.n_kv_heads,
            attn_lora, text_mod.apply_rotary_emb,
        )
        l_mlp = text_mod.mlp(l_in, block.mlp, lora=mlp_lora)
        hidden = hidden + l_attn + l_mlp

    return hidden


def compute_lm_logits(model, text_mod, lora_dict, full_embeds, bidir_len, device):
    """Full-sequence, gradient-enabled forward pass through the text decoder
    with the given LoRA dict applied (pass lora_dict=None for zero-shot),
    returning logits at every position (unlike text.py:lm_head, which only
    returns the last position - built for generation, not training)."""
    inner = inner_model(model)
    text_w = inner.text
    text_cfg = inner.config.text

    hidden = _produce_hidden_with_lora(text_mod, full_embeds, text_w, text_cfg, lora_dict, bidir_len)
    hidden = text_mod.layer_norm(hidden, text_w.post_ln)
    logits = text_w.lm_head(hidden)
    return logits


def build_prompt_embeds(model, text_mod, tokenizer, image, question, device):
    """bos + image + templates['query']['prefix'] + question + suffix, as a
    single embed tensor - the shared prefix for both training (answer tokens
    appended after) and generation (decoded token-by-token after).

    Returns (prompt_embeds, bidir_len) where bidir_len = bos+image token
    count - the span that gets bidirectional attention in compute_lm_logits,
    matching text.py:_produce_hidden's hardcoded mask (which uses the fixed
    TextConfig.prefix_attn=730 constant; we use the actual computed bos+image
    length instead so this stays correct even if that ever varies)."""
    inner = inner_model(model)
    tok_cfg = inner.config.tokenizer

    # NOTE: torch.no_grad() (not inference_mode()) - tensors made under
    # inference_mode() are flagged "inference tensors" and PyTorch refuses to
    # save them for backward later, which is exactly what happens once this
    # embed is used as the `x` in a LoRA-adapted F.linear(x, A) call inside
    # the attention block. no_grad() gives a normal, backward-safe constant.
    with torch.no_grad():
        img_emb = inner._run_vision_encoder(image)
    img_emb = img_emb.detach()

    bos_ids = torch.tensor([[tok_cfg.bos_id]], device=device)
    bos_emb = text_mod.text_encoder(bos_ids, inner.text).detach()

    bidir_len = bos_emb.size(1) + img_emb.size(0)

    prefix_ids = tok_cfg.templates["query"]["prefix"]
    suffix_ids = tok_cfg.templates["query"]["suffix"]
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    prompt_ids = prefix_ids + question_ids + suffix_ids
    prompt_ids_t = torch.tensor([prompt_ids], device=device)
    prompt_emb = text_mod.text_encoder(prompt_ids_t, inner.text).detach()

    prompt_embeds = torch.cat([bos_emb, img_emb[None], prompt_emb], dim=1)
    return prompt_embeds, bidir_len


def build_train_embeds(model, text_mod, tokenizer, image, question, answer, device):
    """Prompt embeds + answer (+eos) tokens, plus labels with -100 masking
    every position except the answer tokens, for teacher-forced training."""
    inner = inner_model(model)
    tok_cfg = inner.config.tokenizer

    prompt_embeds, bidir_len = build_prompt_embeds(model, text_mod, tokenizer, image, question, device)
    n_prefix = prompt_embeds.size(1)

    eos_id = getattr(tok_cfg, "eos_id", None)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]
    answer_ids_t = torch.tensor([answer_ids], device=device)
    answer_emb = text_mod.text_encoder(answer_ids_t, inner.text).detach()

    full_embeds = torch.cat([prompt_embeds, answer_emb], dim=1)
    n_answer = answer_emb.size(1)
    seq_len = n_prefix + n_answer

    labels = torch.full((seq_len,), -100, dtype=torch.long, device=device)
    labels[n_prefix : n_prefix + n_answer] = answer_ids_t[0]

    return full_embeds, labels, seq_len, bidir_len


@torch.inference_mode()
def generate_with_lora(model, text_mod, tokenizer, lora_dict, image, question, device, max_new_tokens=32):
    """Greedy decode using our custom lora dict. No KV cache - recomputes the
    full sequence each step, which is fine for short VQA-style answers on a
    300-example eval/serving set but is not how you'd want to scale this up.
    """
    inner = inner_model(model)
    tok_cfg = inner.config.tokenizer
    eos_id = getattr(tok_cfg, "eos_id", None)

    embeds, bidir_len = build_prompt_embeds(model, text_mod, tokenizer, image, question, device)
    generated_ids = []
    for _ in range(max_new_tokens):
        logits = compute_lm_logits(model, text_mod, lora_dict, embeds, bidir_len, device)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated_ids.append(next_id)
        next_emb = text_mod.text_encoder(torch.tensor([[next_id]], device=device), inner.text)
        embeds = torch.cat([embeds, next_emb], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
