# CENG 467 — Text-to-SQL Semantic Parsing

> **Course:** CENG 467 – Natural Language Understanding and Generation  
> **University:** Izmir Institute of Technology (IYTE), Spring 2026  
> **Project #12:** Text-to-SQL Semantic Parsing  

## Project Description

This project implements a **Text-to-SQL semantic parsing system** that converts natural language questions into executable SQL queries. We use the **Spider benchmark** dataset and explore both prompting-based and fine-tuning-based approaches.

### Task
- **Input:** Natural language question + database schema
- **Output:** Executable SQL query

### Approaches
| Approach | Model | Method |
|---|---|---|
| Baseline 1 | `mistral-small-latest` | Zero-shot prompting |
| Baseline 2 | `mistral-small-latest` | Few-shot prompting (3-shot) |
| Fine-tuned | `Mistral-7B-v0.1` | LoRA instruction fine-tuning (SFT) |
| Fine-tuned | `Qwen2.5-Coder-1.5B` | LoRA instruction fine-tuning (SFT) |

---

## Repository Structure

```
CENG467-Text-to-SQL/
├── src/
│   ├── data_prep.py            # Dataset download, exploration & preprocessing
│   ├── evaluate_baseline.py    # Mistral API baseline evaluation (zero-shot & few-shot)
│   ├── evaluate_mistral.py     # Fine-tuned Mistral-7B LoRA evaluation pipeline
│   ├── evaluate_qwen.py        # Fine-tuned Qwen2.5-Coder-1.5B LoRA evaluation pipeline
│   ├── train_mistral.py        # Mistral-7B LoRA fine-tuning script (Colab/GPU)
│   ├── train_qwen.py           # Qwen2.5-Coder-1.5B LoRA fine-tuning script
│   ├── merge.py                # Merge LoRA adapters into the base Qwen model
│   ├── app.py                  # Gradio demo app (local Text-to-SQL assistant)
│   └── utils.py                # Shared utilities: schema extraction, prompt builders,
│                               #   SQL normalization, metrics, Mistral API wrapper
├── data/
│   ├── database/               # Spider SQLite databases          [not committed]
│   ├── tables.json             # Spider official schema metadata
│   ├── train_formatted.jsonl   # Alpaca-format JSONL for fine-tuning (~10.5 MB) [not committed]
│   └── dataset_stats.json      # Dataset statistics
├── models_02.06.2026/          # Trained LoRA adapter weights     [not committed]
├── results/                    # Evaluation output files          [not committed]
├── notebooks/
│   ├── Dataset_Exploration.ipynb           # EDA on Spider dataset
│   ├── Baseline_Error_Analysis.ipynb       # Qualitative error analysis of baselines
│   └── train_and_evaluate_colab.ipynb      # Full training + evaluation on Google Colab
├── report/                     # LaTeX progress report (LNCS format)
├── requirements.txt            # Python dependencies
├── .env                        # API keys                         [not committed]
└── README.md
```

> **⚠️ Not committed to GitHub** (excluded via `.gitignore`):
> `data/database/`, `data/train_formatted.jsonl`, `models_02.06.2026/`, `results/`, and `.env`
> are omitted due to file size or sensitivity.
> Run `python src/data_prep.py` to regenerate data files and use the training scripts to reproduce model weights.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Aysenursvs/CENG467-Text-to-SQL.git
cd CENG467-Text-to-SQL
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Fine-tuning scripts (`train_mistral.py`, `train_qwen.py`) require a CUDA-capable GPU with at least 4 GB VRAM. The Gradio demo and Qwen evaluation also require CUDA. You can use [Google Colab](https://colab.research.google.com/) for free GPU access.

### 3. Configure API key (for Mistral API baselines only)
Create a `.env` file in the project root:
```
MISTRAL_API_KEY="your_mistral_api_key_here"
```
Get your API key from: https://console.mistral.ai/

---

## Running Experiments

### Data Preparation
```bash
python src/data_prep.py
```
Downloads the Spider dataset, explores its structure, generates `data/train_formatted.jsonl` for fine-tuning, and tests the API connection.

---

### Baseline Evaluation (Mistral API — Zero-shot & Few-shot)
```bash
# Zero-shot and few-shot evaluation with 100 samples using Format B schema
python src/evaluate_baseline.py --num_samples 100 --schema_format format_b

# Options:
#   --num_samples N       Number of validation samples (default: 100)
#   --schema_format FMT   format_a | format_b | format_c (default: format_b)
#   --model MODEL         Mistral model name (default: mistral-small-latest)
#   --delay SECONDS       Delay between API calls (default: 1.5)
```

Results are saved to `results/baseline_results_with_f1.json`.

---

### Fine-Tuning

#### Mistral-7B (Colab recommended)
```bash
python src/train_mistral.py
```
- Base model: `mistralai/Mistral-7B-v0.1`
- Method: LoRA (r=8, α=16) with 4-bit NF4 quantization
- Trainer: `SFTTrainer` (Alpaca-style prompt format)
- Output: saved to `models/sql-mistral-lora/` (or Google Drive path)

#### Qwen2.5-Coder-1.5B (local GPU)
```bash
python src/train_qwen.py
```
- Base model: `Qwen/Qwen2.5-Coder-1.5B`
- Method: LoRA (r=8, α=16) with 4-bit NF4 quantization
- Trainer: `SFTTrainer` with paged AdamW 8-bit optimizer
- Output: saved to `models/sql-qwen-coder-1.5b-lora/`

#### Merge LoRA Adapters (Qwen)
After training, merge the LoRA adapter into the base model for deployment:
```bash
python src/merge.py
```
Output is saved to `models/sql-qwen-coder-1.5b-merged/`.

---

### Fine-Tuned Model Evaluation

#### Evaluate fine-tuned Mistral-7B
```bash
python src/evaluate_mistral.py --num_samples 100
```

#### Evaluate fine-tuned Qwen2.5-Coder-1.5B
```bash
python src/evaluate_qwen.py
```
Reports Execution Accuracy (EX), Exact Match (EM), and Valid SQL Rate on 100 Spider validation samples.

---

### Gradio Demo App (Local)
Run an interactive Text-to-SQL assistant powered by the merged Qwen model:
```bash
python src/app.py
```
Requires the merged model at `models/sql-qwen-coder-1.5b-merged/` and a CUDA GPU.

---

## Schema Serialization Formats

Three formats are implemented in `utils.py` for schema encoding experiments:

| Format | Description | Example |
|---|---|---|
| `format_a` | Plain text listing | `Table: student \| Columns: stuid, fname, age` |
| `format_b` | SQL-style CREATE TABLE | `CREATE TABLE student (stuid INT PRIMARY KEY, ...)` |
| `format_c` | Compact token-efficient | `student(stuid[PK], fname, age)` |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Exact Match (EM)** | Fraction of predicted SQL queries that exactly match the gold SQL after normalization (lowercase, remove aliases, normalize whitespace) |
| **Execution Accuracy (EX)** | Fraction of queries that return the same result set as the gold SQL when executed against the Spider SQLite databases |
| **Precision / Recall / F1** | Set-based IR metrics computed from the intersection of predicted and gold result sets |
| **Valid SQL Rate** | Fraction of predicted queries that execute without errors |

---

## Baseline Results (Mistral API — 100 Validation Samples)

| Method | EM (%) | EX (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|---|
| Zero-shot (Format B) | 21.0 | 79.0 | 82.1 | 84.0 | 82.5 |
| Few-shot / 3-shot (Format B) | 27.0 | 78.0 | 80.8 | 83.0 | 81.3 |

> Results from `results/baseline_results_with_f1.json` (model: `mistral-small-latest`, 100 samples, schema: `format_b`).

---

## Dataset

- **Spider:** 8,659 training + 1,034 validation examples across 200+ databases
- **Source:** [xlangai/spider on HuggingFace](https://huggingface.co/datasets/xlangai/spider)
- **Schema metadata:** `data/tables.json` (Spider official)
- **Fine-tuning data:** `data/train_formatted.jsonl` (Alpaca-style instruction format)

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `transformers` | 4.41.2 | Model loading and inference |
| `peft` | 0.11.1 | LoRA adapter training |
| `trl` | 0.8.6 | SFTTrainer for supervised fine-tuning |
| `bitsandbytes` | 0.43.1 | 4-bit NF4 quantization |
| `datasets` | 2.19.0 | HuggingFace dataset loading |
| `accelerate` | 0.31.0 | Distributed training utilities |
| `openai` | ≥1.0.0 | Mistral API client (OpenAI-compatible) |
| `gradio` | — | Interactive demo interface |
| `sqlparse` | ≥0.5.0 | SQL parsing and normalization |

---

## Team
- Ayşenur Sivaslıgil
- Mustafa Erkoca

## License
This project is for educational purposes as part of CENG 467 coursework at IYTE.
