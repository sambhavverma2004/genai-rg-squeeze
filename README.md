# RG-Squeeze: Retention-Guided Rank Annealing for LoRA

Project built on **LoRA-Squeeze** (Vulić, Grycner, de Laroussilhe, Pfeiffer — Google DeepMind, arXiv 2602.10993, Feb 2026).

## What is here
| path | purpose |
|---|---|
| `rgsqueeze/lora.py` | LoRA layer with exact, memory-efficient rank truncation (QR + r×r SVD core, never forms dense ΔW) |
| `rgsqueeze/policies.py` | Baselines (direct, post, cont, in_std, in_min) and **RG-Squeeze** (`rg`, ours) |
| `rgsqueeze/train.py` | Policy-agnostic loop; identical optimizer-reset + LR re-warm after every squeeze for all policies |
| `llm_experiment.py` | Real LLM runs (HF models; BoolQ / ARC / OpenBookQA / GSM8K) |
| `toy_study.py` | CPU sanity study with controlled spectra (mechanism check, **not** a benchmark) |
| `tests/` | `test_squeeze.py` (7 unit tests), `smoke_llm.py` (end-to-end on a tiny random Llama) |
| `run_all.sh`, `make_report.py` | Core experiment grid + markdown table |

## Run
```bash
pip install torch transformers datasets accelerate
python tests/test_squeeze.py && python tests/smoke_llm.py        # correctness (CPU, seconds)
python llm_experiment.py --model Qwen/Qwen2.5-1.5B --task boolq --policy rg --r-src 64 --r-tgt 4 --steps 200  # time one run first
./run_all.sh                                                      # core grid, then make_report.py prints the table
```

## Honest status
* Unit tests, smoke test and toy study were run. The Hugging Face dataset loaders and the real-LLM runs were **not**
  executable in the authoring sandbox (no Hugging Face access) - verify the first run on your GPU (print one example per task).
* LR for the LLM grid defaults to 2e-4 (a common LoRA value); tune per policy on a held-out slice if time allows.
* Optimizer state: LoRA-Squeeze does not specify what happens to optimizer state at a squeeze. Here every squeeze policy
  gets a fresh AdamW and a 50-step LR re-warm, so comparisons are fair but absolute numbers differ from the paper (Adafactor).
