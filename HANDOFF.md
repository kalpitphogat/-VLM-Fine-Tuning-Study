# Handoff / Continuation Guide

**Read this file first, top to bottom, before touching anything else in this repo.**
It exists so whoever picks this project up next does not have to re-derive a full evening
of environment and architecture debugging. Follow it in order. Everything referenced here
(`config.yaml`, `scripts/*.py`) already exists in this repo.

## 1. What this project is

QLoRA-style ablation study: fine-tune Moondream2 (a 2B-param vision-language model) on a
VQAv2 slice, compare LoRA rank {4,8,16,32}, LR schedule {cosine, constant}, and data
filtering {filtered, unfiltered} against a zero-shot baseline, then serve the best adapter
via FastAPI. Target hardware: a single 16GB GPU (fully run and completed on an RTX 4060 Ti server).
Full technical background is in `VLM_Finetuning_Study_Interview_Prep.docx` in this same
folder — read that if you want the "why" behind every design choice.

## 2. Exact status as of this point — COMPLETED & VERIFIED

- **End-to-End Ablation Study Completed:** The full pipeline (`ablation.py` -> `baseline.py` -> `train.py` -> `evaluate.py` -> `plots.py`) has been run successfully.
- **Key Findings & Discoveries:**
  - **Dataset Swap:** The original `HuggingFaceM4/VQAv2` dataset streams images from the official COCO servers, which repeatedly failed due to connection resets. We successfully swapped to the `lmms-lab/VQAv2` mirror, which embeds images directly in Parquets on HuggingFace LFS.
  - **Dtype Alignment:** The base model natively loads in `bfloat16` on compatible hardware (like RTX 4060 Ti). We patched the scripts to dynamically align the LoRA parameters to the model's actual parameter dtype to prevent precision mismatch errors.
  - **Unfiltered Data Victory:** The biggest finding is that keeping "trivial" yes/no and single-digit/character answers in the training pool is crucial. Filtering them out left only complex multi-word answers that 3 epochs of training could not learn, keeping all "filtered" runs at baseline (2.33%). The unfiltered dataset gave the model easy reference points, raising accuracy to **25.67%**.
  - **Optimal Rank:** Within the filtered pool, increasing rank from 4 to 8, 16, or 32 did not yield accuracy gains; rank 4 (0.16% trainable parameters) is the most compute-efficient sweet spot.
  - **LR Schedule:** Cosine vs. Constant schedules performed identically (2.33%) at rank 4.
- **FastAPI Serving Server Live:** The best adapter (`rank4_cosine_unfiltered`) is serving live on the remote GPU server (FastAPI + Uvicorn) on port 8000. You can query it via an SSH tunnel on port 7860.
- **All Plots Generated:** Saved in `results/accuracy_vs_rank.png` and `results/filtered_vs_unfiltered.png`.

## 3. Exact Kaggle environment setup (do this in a brand-new notebook)

These exact pins were only arrived at after a long trial-and-error session. **Do them in
this order, in one notebook, without restarting the session in between** (Kaggle's
"Restart Session" reprovisions the whole container and wipes pip installs — it is NOT like
a normal Jupyter kernel restart; if you must clear a stale import in the same process, use
`del sys.modules[...]`, not Restart Session).

1. New Notebook → Settings → **GPU T4**, **Internet: On**.
2. Upload this repo's contents as a Kaggle Dataset (zip it, upload, add as notebook input),
   then:
   ```python
   !cp -r /kaggle/input/<your-dataset-name>/vlm-lora-study /kaggle/working/
   %cd /kaggle/working/vlm-lora-study
   ```
3. Install pinned core deps, in exactly this order, in ONE cell:
   ```python
   !pip install -q --force-reinstall "transformers==4.46.3" "tokenizers==0.20.3"
   !pip uninstall -y -q tensorflow scipy
   !pip install -q "datasets==2.20.0"
   !pip install -q peft pillow pyyaml pandas matplotlib fastapi "uvicorn[standard]" python-multipart einops wandb tqdm bitsandbytes
   ```
   **Why each of these, briefly** (full detail in the interview-prep docx if needed):
   - `transformers==4.46.3` / `tokenizers==0.20.3`: Kaggle's default transformers (5.0.0) has
     an internal tied-weights refactor (`all_tied_weights_keys`) that Moondream2's custom
     `trust_remote_code` modeling class predates and crashes against. This exact pair is the
     confirmed-working combination.
   - Removing `tensorflow`/`scipy`: transformers 4.46.3 eagerly imports both at package-load
     time for an unrelated object-detection loss registry, and this specific Kaggle image has
     a broken numpy/scipy pairing that only surfaces through that import chain. We don't need
     either package (PyTorch-only project), so removing them makes transformers skip those
     code paths entirely via its own `is_tf_available()`/`is_scipy_available()` guards. Do NOT
     try to fix numpy/scipy directly — that was tried repeatedly and made things worse.
   - `datasets==2.20.0`: newer `datasets` (5.0.0) dropped support for the legacy "loading
     script" format that `HuggingFaceM4/VQAv2` still uses on the Hub.
4. Verify the pin actually took (transformers has a habit of getting pre-imported by Kaggle's
   kernel startup before your install runs, which silently defeats the pin):
   ```python
   import transformers
   print(transformers.__version__)   # must print 4.46.3, not 5.0.0
   ```
   If it prints the wrong version, run:
   ```python
   import sys
   for m in list(sys.modules):
       if m == "transformers" or m.startswith("transformers."):
           del sys.modules[m]
   import transformers
   print(transformers.__version__)
   ```
5. Sanity-load the model once before running any repo script:
   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained(
       "vikhyatk/moondream2", trust_remote_code=True, torch_dtype=torch.float16,
   ).to("cuda")
   tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
   print("loaded ok")
   ```
   Note: `torch_dtype=`, not `dtype=` — the newer kwarg name doesn't exist in this pinned
   version.

If this cell sequence doesn't work anymore (e.g. Kaggle's base image has moved on), don't
start randomly upgrading/downgrading packages one at a time — that's what cost the most time
this session. Re-derive pins methodically: check `pip show transformers`, read the actual
traceback's innermost frame, and change one thing at a time.

## 4. Exact commands to run from here, in order

```bash
# 1. Confirm/finish data prep (may already be done - check data/*.jsonl first)
python scripts/prepare_data.py

# 2. Zero-shot baseline (records the number everything else is compared against)
python scripts/baseline.py

# 3. THE CRITICAL GATE - do not skip, do not proceed past a failure
python scripts/train.py --smoke_test
```

**Stop here if step 3 fails.** Do not run `ablation.py` until `--smoke_test` prints
`Smoke test passed.` — see §5 for what to do if it fails.

```bash
# 4. Full ablation matrix (long-running: several fine-tunes back to back)
python scripts/ablation.py

# 5. Plots
python results/plots.py

# 6. Look at the results
cat results/metrics.csv
```

Then fill in the README's "Results" table and "Findings" section with the real numbers
(the README currently has placeholder tables marked with `*To be written*`).

## 5. If `train.py --smoke_test` fails

This is the single riskiest part of the whole project, so it gets its own section. Moondream2
does not expose a standard HF training interface (no usable `forward(labels=...)`), so
`scripts/common.py` hand-rolls the training forward pass by calling the model's own internal
functions directly, with modifications. This was reconstructed by reading Moondream2's live
source on a real GPU, not guessed from documentation.

**Step 1**: run `python scripts/inspect_model.py`. It dumps the live module tree and the
actual source of `text_decoder`, `attn`, `mlp`, `_produce_hidden`, `lm_head`, `text_encoder`
from whichever Moondream2 revision you're actually running. Compare that dump against the
assumptions documented in `scripts/common.py`'s docstrings (each function that depends on
this has a comment citing exactly what it assumes and why).

**Step 2**: the specific things `common.py` assumes, in case Moondream2 has been updated
since and one of these no longer holds:
- `model.model.text.blocks[i].attn` / `.mlp` are `ModuleDict`s of real `nn.Linear` layers
  named `qkv`, `proj`, `fc1`, `fc2` (not classes with their own `forward()`).
- The real `attn()`/`text_decoder()` always mutate a persistent `block.kv_cache` (built for
  generation, wrong for training) — `common.py`'s `_attn_no_cache_with_lora` is a from-scratch
  copy of `attn()`'s math with that cache dependency removed.
- `TextConfig.prefix_attn` (730) represents BOS(1) + image tokens(729) getting bidirectional
  (not causal) attention, with causal masking after — `common.py` uses the *actually computed*
  BOS+image length rather than trusting this constant, specifically so it keeps working if
  that count ever changes.
- Moondream2's own native LoRA hook is a nested dict:
  `lora["text"]["blocks"][str(layer_idx)]["attn"|"mlp"]["qkv"|"proj"|"fc1"|"fc2"]["A"|"B"]`
  — `MoondreamLoRA` in `common.py` builds exactly this structure as real `nn.Parameter`s.

**Step 3**: if the smoke test says gradients aren't reaching the LoRA parameters, or the
loss/shapes look wrong, start by re-reading `inspect_model.py`'s dump of `attn`/`text_decoder`
source side-by-side with `common.py`'s `_attn_no_cache_with_lora`/`_produce_hidden_with_lora`
— the bug is almost certainly a mismatch between those two, not somewhere else in the repo.

## 6. Key files, ranked by how much you need to understand before changing them

1. `scripts/common.py` — the risky, hand-rolled part (LoRA + forward pass). Read the module
   docstring and every function docstring before editing.
2. `config.yaml` — every hyperparameter and path. No script hardcodes anything; if you need
   to change a hyperparameter, change it here, not in a script.
3. `scripts/train.py` — the training loop; `--smoke_test` lives here.
4. `scripts/inspect_model.py` — your debugging tool if anything about the model's architecture
   has changed since this pause point.
5. Everything else (`prepare_data.py`, `baseline.py`, `evaluate.py`, `ablation.py`, `serve.py`,
   `results/plots.py`) is comparatively low-risk, ordinary glue code.

## 7. Don't re-litigate these decisions without a reason

- **fp16, not 4-bit quantization** (`config.yaml: quantization.load_in_4bit: false`) —
  unverified whether Moondream2's custom loading path even honors `BitsAndBytesConfig`, and
  fp16 fits a 16GB GPU fine since the base model is frozen (no optimizer state). Flip it on
  only after confirming via `inspect_model.py` that parameter dtypes actually change.
- **Hand-rolled LoRA, not `peft.get_peft_model()`** — see §5; Moondream2 has its own native
  LoRA mechanism this project builds against directly.
- **Subprocess-per-run in `ablation.py`**, not in-process looping — GPU memory isolation on a
  16GB card across several back-to-back fine-tunes.
