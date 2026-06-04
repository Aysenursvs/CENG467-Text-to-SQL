import os
import re
import sqlite3
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ─── Configuration ────────────────────────────────────────────────────────────
ACTIVE_MODEL   = "qwen-coder-1.5b"
MODEL_CONFIGS  = {"qwen-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B"}
BASE_MODEL     = MODEL_CONFIGS[ACTIVE_MODEL]
LORA_MODEL_DIR = f"models/sql-{ACTIVE_MODEL}-lora"
DB_DIR         = "data/database"
TEST_COUNT     = 100    
MAX_NEW_TOKENS = 150    
MAX_INPUT_LEN  = 512
NUM_BEAMS      = 4      

# ─── Database Helpers ─────────────────────────────────────────────────────────
def get_db_schema(db_id: str) -> str:
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return ""
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return "\n\n".join(t[0] for t in tables if t[0])
    except Exception:
        return ""

def execute_sql(db_id: str, sql_query: str):
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    try:
        conn   = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception:
        return "ERROR"

def normalize_result(result) -> set:
    if result == "ERROR":
        return "ERROR"

    normalized = set()
    for row in result:
        norm_row = []
        for val in row:
            if val is None:
                norm_row.append("")
            elif isinstance(val, str):
                norm_row.append(val.strip().lower())
            elif isinstance(val, float):
                norm_row.append(round(val, 4))
            else:
                norm_row.append(val)
        normalized.add(tuple(norm_row))
    return normalized

def calculate_exact_match(predicted_sql: str, target_sql: str) -> bool:
    """Basic normalization to check if the SQL strings match exactly."""
    pred_clean = re.sub(r"\s+", " ", predicted_sql).strip().lower()
    targ_clean = re.sub(r"\s+", " ", target_sql).strip().lower()
    # Remove trailing semicolons for fair comparison
    pred_clean = pred_clean.rstrip(";")
    targ_clean = targ_clean.rstrip(";")
    return pred_clean == targ_clean

# ─── SQL Extraction ───────────────────────────────────────────────────────────
def extract_sql(raw_output: str) -> str:
    text = raw_output.strip()

    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*",    "", text)
    text = re.sub(r"--[^\n]*", "", text)

    match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if match:
        text = text[match.start():]

    text = text.split(";")[0]
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace('"', "'")

    return text

# ─── Prompt Builder ───────────────────────────────────────────────────────────
def build_prompt(schema: str, question: str) -> str:
    return (
        "### Instruction:\n"
        "Convert the natural language question into a SQL query based on the provided database schema.\n"
        "Rules:\n"
        "1. Use ONLY table and column names explicitly defined in the schema.\n"
        "2. Output ONLY the raw SQL query — no explanation, no markdown fences.\n"
        "3. Use JOIN only when columns from multiple tables are required.\n\n"
        "### Input:\n"
        f"Schema:\n{schema}\n"
        f"Question: {question}\n\n"
        "### Response:\n"
    )

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  Evaluation: {BASE_MODEL} + LoRA")
    print(f"{'='*60}\n")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — check your driver / CUDA toolkit.")

    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Beam size    : {NUM_BEAMS}")
    print(f"  Max new tok  : {MAX_NEW_TOKENS}\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("  Loading base model (4-bit NF4)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Attaching LoRA adapter from '{LORA_MODEL_DIR}'...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    model.eval()
    print("  ✅ Model ready.\n")

    val_dataset = load_dataset("xlangai/spider", split="validation")
    n_samples   = TEST_COUNT if TEST_COUNT else len(val_dataset)
    print(f"  Evaluating {n_samples} / {len(val_dataset)} validation examples.\n")

    exec_correct = 0
    exact_match  = 0
    valid_sql    = 0
    skipped      = 0
    
    start_time = time.time()

    for i in range(n_samples):
        sample     = val_dataset[i]
        db_id      = sample["db_id"]
        question   = sample["question"]
        target_sql = sample["query"]
        schema     = get_db_schema(db_id)

        if not schema:
            skipped += 1
            print(f"[{i+1:4d}/{n_samples}] SKIPPED — missing db: {db_id}")
            continue

        prompt = build_prompt(schema, question)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=NUM_BEAMS,
                early_stopping=True,
                repetition_penalty=1.1,
            )

        new_tokens    = outputs[0][inputs.input_ids.shape[-1]:]
        raw_output    = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predicted_sql = extract_sql(raw_output)

        # ── Metric Evaluations ──────────────────────────────────────────
        target_result    = normalize_result(execute_sql(db_id, target_sql))
        predicted_result = normalize_result(execute_sql(db_id, predicted_sql))

        is_valid_sql = predicted_result != "ERROR"
        is_exec_correct = (target_result == predicted_result) and is_valid_sql
        is_exact_match = calculate_exact_match(predicted_sql, target_sql)

        if is_valid_sql:
            valid_sql += 1
            
        if is_exact_match:
            exact_match += 1

        if is_exec_correct:
            exec_correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
            print(f"\n  ── Error Analysis [{i+1}] ────────────────────────────")
            print(f"  Question  : {question}")
            print(f"  Target SQL: {target_sql}")
            print(f"  Model SQL : {predicted_sql}")
            if not is_valid_sql:
                print(f"  Note      : model SQL caused a syntax/execution error")
            elif is_exact_match and not is_exec_correct:
                print(f"  Note      : Exact match achieved, but execution failed (database state issue)")
            print(f"  {'─'*50}\n")

        elapsed   = time.time() - start_time
        evaluated = i + 1 - skipped
        avg_sec   = elapsed / evaluated if evaluated > 0 else 0
        ex_so_far = exec_correct / evaluated * 100 if evaluated > 0 else 0
        
        print(
            f"[{i+1:4d}/{n_samples}] {status} | "
            f"db: {db_id:<15} | "
            f"EX: {ex_so_far:5.1f}% | "
            f"{avg_sec:.1f}s/sample"
        )

    # ── Final Report ───────────────────────────────────────────────────────
    evaluated  = n_samples - skipped
    ex_score   = (exec_correct / evaluated * 100) if evaluated > 0 else 0.0
    em_score   = (exact_match / evaluated * 100)  if evaluated > 0 else 0.0
    vsr_score  = (valid_sql / evaluated * 100)    if evaluated > 0 else 0.0
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION REPORT — {BASE_MODEL} + LoRA")
    print(f"{'='*60}")
    print(f"  Evaluated          : {evaluated}  (Skipped: {skipped})")
    print(f"  Execution Accuracy : {ex_score:.2f}% ({exec_correct}/{evaluated})")
    print(f"  Exact Match (EM)   : {em_score:.2f}% ({exact_match}/{evaluated})")
    print(f"  Valid SQL Rate     : {vsr_score:.2f}% ({valid_sql}/{evaluated})")
    print(f"  Total Time         : {total_time / 60:.1f} min")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()