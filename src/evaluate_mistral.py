"""
evaluate_mistral.py

Evaluate a fine-tuned Mistral Text-to-SQL model on the Spider validation
dataset using:

- Exact Match (EM)
- Execution Accuracy (EX)

The script loads a LoRA-adapted Mistral model, generates SQL queries
for Spider validation questions, executes both predicted and ground-truth
queries, and reports evaluation metrics.
"""

import os
import sqlite3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

from utils import extract_schema_from_sample, serialize_schema_format_b, normalize_sql

# --- CONFIGURATION ---
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL_DIR = "/content/drive/MyDrive/sql-mistral-lora/checkpoint-175"
DB_DIR = "data/database"


def build_prompt(db_id, schema_text, question):
    """
    Construct the instruction prompt used for SQL generation.
    """
    return (
        "### Instruction:\n"
        "You are an expert SQL developer. Your task is to translate the given natural language question into a valid executable SQL query.\n\n"
        "### Input:\n"
        f"Database: {db_id}\nSchema:\n{schema_text}\n\nQuestion: {question}\n\n"
        "### Response:\n"
    )

def execute_sql(db_id, sql_query):
    """
    Execute a SQL query against the SQLite database associated with db_id.

    Returns:
        set: Query results as a set of tuples.
        str: Error message if execution fails.
    """
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    
    if not os.path.exists(db_path):
        return "DB_NOT_FOUND"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        return set(result) 
    except Exception as e:
        return f"SQL_ERROR: {e}"

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Text-to-SQL model on Spider validation set")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of validation samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate for SQL")
    parser.add_argument("--schema_format", type=str, default="format_b", choices=["format_b"], help="Schema serialization format")
    args = parser.parse_args()

    print("=" * 70)
    print("Loading Text-to-SQL evaluation pipeline...")
    print("=" * 70)

    # 4-bit quantization to reduce GPU memory usage.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print("[2/3] Loading base model (Mistral-7B) in 4-bit...")
    # Automatically distribute model layers across available devices.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto" 
    )

    print("[3/3] Loading fine-tuned LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    model.eval()
    
    print("\n✅ Model loaded successfully. Starting evaluation on Spider validation set...\n")

    dataset = load_dataset("xlangai/spider")
    val_data = dataset["validation"]
    eval_samples = min(args.num_samples, len(val_data))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    exact_matches = 0
    exec_matches = 0
    exec_total = 0

    for i in range(eval_samples):
        sample = val_data[i]
        db_id = sample["db_id"]
        question = sample["question"]
        target_sql = sample["query"]

        schema_info = extract_schema_from_sample(sample)
        schema_text = serialize_schema_format_b(schema_info)

        prompt = build_prompt(db_id, schema_text, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate SQL query deterministically for evaluation.
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the generated SQL from the model response.
        generated_sql = generated_text.split("### Response:")[-1].strip()
        if ";" in generated_sql:
            generated_sql = generated_sql.split(";")[0].strip() + ";"

        print(f"\nQuestion: {question}")
        print(f"Target SQL: {target_sql}")
        print(f"Predicted SQL: {generated_sql}")
        print("-" * 60)

        em = 1 if normalize_sql(generated_sql) == normalize_sql(target_sql) else 0
        exact_matches += em

        gold_result = execute_sql(db_id, target_sql)
        pred_result = execute_sql(db_id, generated_sql)
        if "SQL_ERROR" not in str(gold_result) and "SQL_ERROR" not in str(pred_result) and gold_result != "DB_NOT_FOUND" and pred_result != "DB_NOT_FOUND":
            exec_total += 1
            if gold_result == pred_result:
                exec_matches += 1

        status = "✅" if em == 1 else "❌"
        print(
            f"[{i + 1:3d}/{eval_samples}] {status} EM={exact_matches}/{i + 1} "
            f"| DB={db_id} | {question[:60]}..."
        )

    em_score = (exact_matches / eval_samples) * 100 if eval_samples > 0 else 0
    ex_score = (exec_matches / exec_total) * 100 if exec_total > 0 else 0

    print("\n" + "=" * 60)
    print(f"Exact Match: {exact_matches}/{eval_samples} = {em_score:.2f}%")
    print(f"Execution Accuracy: {exec_matches}/{exec_total} = {ex_score:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()