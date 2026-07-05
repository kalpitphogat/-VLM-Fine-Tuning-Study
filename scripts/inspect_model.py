"""Run this FIRST on your Kaggle T4 notebook, before trusting train.py.

Moondream2 ships custom modeling code (trust_remote_code=True) that this repo
had to reverse-engineer from the published source on huggingface.co/vikhyatk/moondream2
without being able to execute it. Two things in particular need a live check:

  1. Module tree: confirm the model's Linear-like layers are named the way
     config.yaml's `lora.target_modules` documents (qkv, proj, fc1, fc2) and
     that config.yaml's lora.n_layers/dim/ff_dim match TextConfig on the model
     you actually loaded.
  2. Forward path for training: `scripts/train.py` calls into a function
     (currently assumed to be `text_decoder` from text.py, see
     TRAIN_FORWARD_FN below) to get a full-sequence, gradient-enabled hidden
     state so a teacher-forced loss can be computed. This prints that
     function's source so you can eyeball it matches what train.py assumes
     before running a real (expensive) training job. train.py also has a
     `--smoke_test` flag that will fail loudly and fast if the assumption is
     wrong - run that before the full ablation matrix regardless.

Usage (on Kaggle, GPU on):
    python scripts/inspect_model.py
"""
import inspect
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from common import load_config, load_moondream  # noqa: E402


TRAIN_FORWARD_FNS = ["text_decoder", "_produce_hidden", "attn", "lm_head", "text_encoder"]


def main():
    config = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    print("Loading Moondream2 (this downloads ~4GB of weights on first run)...")
    model, tokenizer = load_moondream(config)

    print("\n" + "=" * 88)
    print("FULL MODULE TREE (model.named_modules()) - first 300 entries")
    print("=" * 88)
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 300:
            print("... (truncated)")
            break
        print(f"{name:70s} {module.__class__.__name__}")

    print("\n" + "=" * 88)
    print("LEAF MODULES WHOSE NAME ENDS IN config.yaml lora.target_modules")
    print("=" * 88)
    base_names = set(config["lora"]["target_modules"])
    found_any = False
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in base_names and not list(module.children()):
            found_any = True
            print(f"{name:70s} {module.__class__.__name__}")
    if not found_any:
        print(
            "(none found as real nn.Module submodules - expected for Moondream2, "
            "which applies attn/mlp projections functionally rather than via "
            "nn.Linear submodules. This confirms PEFT's get_peft_model() would "
            "have nothing to attach to; scripts/common.py's MoondreamLoRA hand-"
            "rolled adapter is the intended path, not the module tree above.)"
        )

    print("\n" + "=" * 88)
    print("TextConfig on the loaded model (compare against config.yaml lora.n_layers/dim/ff_dim)")
    print("=" * 88)
    try:
        text_cfg = model.config.text if hasattr(model.config, "text") else model.model.config.text
        print(text_cfg)
    except AttributeError as e:
        print(f"Could not read model.config.text directly ({e}); check model.config manually:")
        print(model.config)

    print("\n" + "=" * 88)
    print("SOURCE OF KEY FUNCTIONS train.py DEPENDS ON")
    print("=" * 88)
    import importlib

    text_mod = None
    for candidate in ("moondream", "text"):
        try:
            text_mod = importlib.import_module(candidate)
            break
        except ImportError:
            continue

    if text_mod is None:
        # trust_remote_code modules are loaded into a dynamically named module;
        # fall back to searching sys.modules for one that defines text_decoder.
        for mod_name, mod in list(sys.modules.items()):
            if hasattr(mod, "text_decoder"):
                text_mod = mod
                break

    if text_mod is None:
        print(
            "Could not locate the dynamically-imported modeling module in sys.modules. "
            "List sys.modules keys containing 'moondream' and import that one manually, e.g.:\n"
            "  import sys; [m for m in sys.modules if 'moondream' in m]"
        )
    else:
        for fn_name in TRAIN_FORWARD_FNS:
            fn = getattr(text_mod, fn_name, None)
            print(f"\n--- {fn_name} " + "-" * (70 - len(fn_name)))
            if fn is None:
                print("(not found in this revision - train.py's assumption is WRONG, fix needed)")
                continue
            try:
                print(inspect.getsource(fn))
            except (OSError, TypeError) as e:
                print(f"(could not get source: {e})")

    print("\n" + "=" * 88)
    print("Next step: run `python scripts/train.py --smoke_test` to do one forward+backward")
    print("pass on a single example and confirm gradients reach the LoRA parameters.")
    print("=" * 88)


if __name__ == "__main__":
    main()
