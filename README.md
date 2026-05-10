# CENG 467 — Text-to-SQL Semantic Parsing

> **Course:** CENG 467 – Natural Language Understanding and Generation  
> **University:** Izmir Institute of Technology (IYTE), Spring 2026  
> **Project #12:** Text-to-SQL Semantic Parsing  

## Project Description

This project implements a Text-to-SQL semantic parsing system that converts natural language questions into executable SQL queries. We use the **Spider benchmark** dataset and evaluate prompting strategies with the **Google Gemini API**.

### Task
- **Input:** Natural language question + database schema
- **Output:** Executable SQL query

### Approach
- **Baseline 1:** Zero-shot prompting (Gemini 2.0 Flash)
- **Baseline 2:** Few-shot prompting (3-shot, Gemini 2.0 Flash)
- **Schema Encoding:** 3 serialization formats (plain text, CREATE TABLE, compact)
- **Planned:** Instruction-tuned fine-tuning for final submission

## Repository Structure

```
CENG467-Text-to-SQL/
├── src/
│   ├── data_prep.py      # Dataset download, exploration, preprocessing
│   ├── evaluate.py        # Baseline evaluation pipeline
│   ├── utils.py           # Schema extraction, prompts, metrics, API wrapper
│   └── train.py           # (Planned) Fine-tuning script
├── data/                  # Dataset statistics and cached data
├── results/               # Evaluation results (JSON)
├── notebooks/             # Exploration notebooks
├── report/                # LaTeX progress report (LNCS format)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
└── README.md
```

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

### 3. Configure API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY="your_gemini_api_key_here"
```
Get your API key from: https://aistudio.google.com/apikey

## Running Experiments

### Data Preparation
```bash
python src/data_prep.py
```
Downloads the Spider dataset, explores its structure, and tests the API connection.

### Baseline Evaluation
```bash
# Evaluate with 50 samples using Format A schema
python src/evaluate.py --num_samples 50 --schema_format format_a

# Evaluate with 100 samples using Format B schema
python src/evaluate.py --num_samples 100 --schema_format format_b

# Options:
#   --num_samples N       Number of validation samples (default: 50)
#   --schema_format FMT   format_a | format_b | format_c (default: format_a)
#   --model MODEL         Gemini model name (default: gemini-2.0-flash)
#   --delay SECONDS       Delay between API calls (default: 1.5)
```

Results are saved to `results/baseline_results.json`.

## Evaluation Metrics
- **Exact Match (EM):** Fraction of predicted SQL queries that exactly match the gold SQL after normalization
- **Execution Accuracy (EX):** (Planned) Fraction of queries returning correct results when executed

## Dataset
- **Spider:** 8,659 training + 1,034 validation examples across 200+ databases
- **Source:** [xlangai/spider on HuggingFace](https://huggingface.co/datasets/xlangai/spider)

## Team
- Ayşenur S
- Mustafa [Soyadı]

## License
This project is for educational purposes as part of CENG 467 coursework at IYTE.
