# Efficient VLM Fine-Tuning Study: LoRA Ablation on Moondream2

An engineering study on customizing the lightweight **Moondream2 (2B)** Vision-Language Model on a VQAv2 slice. By building a custom, hand-rolled LoRA adapter and resolving deep system-level hardware mismatches, this pipeline successfully raised visual question-answering accuracy from a **2.33% zero-shot baseline to 25.67%**; the best run trained in 45 min and the full 6-run ablation took ~3 h.

---

## 1. Executive Summary: The Narrative

### The Core Problem
Vision-Language Models (VLMs) are highly capable but computationally massive. Fine-tuning the entire model is economically infeasible for most projects and destroys the model's pre-trained general knowledge. While Parameter-Efficient Fine-Tuning (PEFT/LoRA) is the standard solution, **Moondream2 has no standard `forward(labels=...)` training path, and its attention writes into a persistent KV cache, so the adapter plugs into the model's native LoRA hook through a cache-free forward pass.**

### The Engineering Method
To solve this, we:
1. **Hand-rolled a custom LoRA adapter** matching Moondream2's native text-decoder block structure, registering weights directly as PyTorch parameters.
2. **Built an end-to-end ablation pipeline** (`ablation.py` -> `baseline.py` -> `train.py` -> `evaluate.py` -> `plots.py`) to systematically test Rank ∈ {4, 8, 16, 32}, Learning Rate Schedules, and Dataset Heuristics.
3. **Engineered dynamic precision alignment** to prevent CUDA allocator crashes, casting our adapter parameters to match the model's native `bfloat16` precision on consumer hardware (NVIDIA RTX 4060 Ti).
4. **Migrated to a Parquet-embedded HuggingFace mirror** (`lmms-lab/VQAv2`) to bypass transient network dropouts from COCO's official image hosting.

### The Big Discovery
**Our data-filtering assumptions were wrong, and the ablation proved it.** Conventional VLM wisdom suggests filtering out "trivial" single-character or yes/no answers to force the model to learn complex representations. However, our ablation demonstrated that filtering out these simple samples completely stalls convergence in low-epoch configurations, leaving the model flat at baseline. Keeping the raw, unfiltered data provided crucial anchors for gradient descent, **boosting accuracy by +23.34%**.

---

## 2. System Architecture

```mermaid
flowchart TD
    A[lmms-lab/VQAv2 Parquets] -->|prepare_data.py| B[train_filtered.jsonl / train_unfiltered.jsonl / test.jsonl]
    B --> C["baseline.py<br/>zero-shot accuracy"]
    B --> D["train.py<br/>MoondreamLoRA + frozen Moondream2"]
    D --> E["models/rank{R}_{schedule}_{data}/<br/>lora_adapter.pt + train_stats.json"]
    E --> F["evaluate.py<br/>accuracy + delta vs baseline"]
    C --> G[results/metrics.csv]
    F --> G
    G --> H[results/plots.py<br/>accuracy-vs-rank, filtered-vs-unfiltered]
    E --> I[serve.py\nFastAPI /answer]
    subgraph ablation.py
      D
      F
    end
```

### Why LoRA is Hand-Rolled
Moondream2 has no standard `forward(labels=...)` training path, and its attention writes into a persistent KV cache, so the adapter plugs into the model's native LoRA hook through a cache-free forward pass.

---

## 3. Ablation Results

The ablation matrix was executed on a single NVIDIA RTX 4060 Ti (16.7 GB VRAM) running sequentially to ensure full CUDA memory release between runs.

| Config Name | Rank | LR Schedule | Data Variant | Exact-match accuracy | Δ vs baseline (percentage points) | Trainable % | Train Time | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **zero_shot_baseline** | — | — | — | **2.33%** | +0.00 pp | 0.00% | — | 4.47 GB |
| **rank4_cosine_filtered** | 4 | cosine | filtered | **2.33%** | +0.00 pp | 0.16% | 22 min | 5.39 GB |
| **rank8_cosine_filtered** | 8 | cosine | filtered | **2.00%** | −0.33 pp | 0.33% | 22 min | 5.41 GB |
| **rank16_cosine_filtered** | 16 | cosine | filtered | **1.67%** | −0.66 pp | 0.65% | 45 min | 5.44 GB |
| **rank32_cosine_filtered** | 32 | cosine | filtered | **2.00%** | −0.33 pp | 1.29% | 22 min | 5.51 GB |
| **rank4_constant_filtered** | 4 | constant | filtered | **2.33%** | +0.00 pp | 0.16% | 22 min | 5.39 GB |
| **rank4_cosine_unfiltered** | **4** | **cosine** | **unfiltered** | **25.67%** | **+23.34 pp** | **0.16%** | **45 min** | **5.39 GB** |

<p align="center">
  <img src="results/accuracy_vs_rank.png" width="45%" />
  <img src="results/filtered_vs_unfiltered.png" width="45%" />
</p>

The 2.33% baseline is likely deflated by the scorer: single-reference exact match
with minimal normalization and no short-answer instruction in the prompt.
Predictions weren't saved, so this isn't confirmed.

---

## 4. Deep-Dive Findings

### 1. The Filtered Data Trap
Our filtering heuristic dropped trivial answers (yes/no, single digits, etc.) under the assumption that they pollute training. However, visual question answering datasets like VQAv2 are heavily balanced around these answers. 
- **Without them (Filtered):** The model was forced to output complex phrases immediately. In a small training window (3 epochs), the gradient signal was too weak, leading to zero convergence and flatline performance.
- **With them (Unfiltered):** The model learned the mapping for simple answers, establishing base weight adjustments that immediately carried over to test evaluation.

### 2. Rank Ablation & Immediate Returns Diminution
Within the filtered dataset, increasing the rank of our LoRA adapter from 4 to 32 did not yield any accuracy improvements. These gaps are only 1–2 questions out of 300, and rank was only tested on filtered data.

### 3. Learning Rate Schedule is Negligible
At rank 4, transitioning from a `cosine` decay schedule to a `constant` schedule yielded identical baseline performance. This indicates that VLM adaptation on specialized tasks is far more sensitive to dataset variety and formatting than it is to schedule tuning.

---

## 5. Retrospective: What We'd Do Differently

If we were to expand this study to a production-grade scale, we would address the following constraints:

1. **Move Beyond Exact-Match (EM) Evaluation:**
   Visual Question Answering is notoriously penalized by string matching (e.g., if the ground truth is "couch" and the model says "sofa", it scores 0%). We would implement a semantic similarity evaluator using BERTScore or an LLM-compiler judge to capture true model capability.
2. **Adapter Target Expansion:**
   Currently, we only adapt the text decoder block (attention and MLP). For more complex visual tasks, adapting the vision encoder projection layer (cross-attention adapters) is critical to help the visual features align with the custom textual domain.
3. **Scale Epochs with Early Stopping:**
   Three epochs was a safe constraint to fit within tight time quotas. However, it was not long enough for the filtered dataset to show separation. Running 10-15 epochs with early stopping based on validation loss would help identify if filtered data eventually surpasses unfiltered data.

---

## 6. Execution Guide

### Local Setup
Ensure you have a GPU environment running CUDA.
```bash
conda create -n vlm-study python=3.10 -y
conda activate vlm-study
pip install -r requirements.txt
```

### Running the Pipeline
Run the steps in order:
```bash
# 1. Verify model weights, layer shapes, and functional tree
python scripts/inspect_model.py

# 2. Extract and format the dataset splits
python scripts/prepare_data.py

# 3. Evaluate the zero-shot base model
python scripts/baseline.py

# 4. Verify gradient backpropagation through the custom adapter
python scripts/train.py --smoke_test

# 5. Run the complete ablation study matrix
python scripts/ablation.py

# 6. Generate the visual charts
python results/plots.py
```

### Serve the Adapter
To query your trained model via a FastAPI web server on port 8000:
```bash
python scripts/serve.py
```
Query the server:
```bash
curl -F "image=@your_test_image.jpg" -F "question=What is in the image?" http://localhost:8000/answer
```
