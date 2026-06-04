"""
data_prep.py - Spider dataset download, inspection, and preprocessing script.

This script:
    1. Downloads the Spider dataset from Hugging Face
    2. Prints dataset structure and sizes
    3. Extracts schema information automatically
    4. Shows three schema serialization formats
    5. Builds zero-shot and few-shot prompt examples
    6. Tests the Mistral API connection
"""

import os
import sys
import json

from datasets import load_dataset

# Import local utils from the src folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    extract_schema_from_sample,
    build_schema_cache,
    serialize_schema_format_a,
    serialize_schema_format_b,
    serialize_schema_format_c,
    create_zero_shot_prompt,
    create_few_shot_prompt,
    get_mistral_client,
    get_sql_prediction,
    calculate_exact_match,
    DEFAULT_FEW_SHOT_EXAMPLES,
)


def main():
    # Step 1: download the Spider dataset.
    print("=" * 60)
    print("STEP 1: Downloading / Loading Spider Dataset...")
    print("=" * 60)
    dataset = load_dataset("xlangai/spider")

    # Show dataset sizes.
    print("\n--- DATASET SIZES ---")
    print(f"  Train set      : {len(dataset['train'])} samples")
    print(f"  Validation set : {len(dataset['validation'])} samples")

    # Inspect the first sample.
    sample = dataset["train"][0]

    print("\n--- FIRST SAMPLE INSPECTION ---")
    print(f"  Database (db_id) : {sample['db_id']}")
    print(f"  Question         : {sample['question']}")
    print(f"  Target SQL       : {sample['query']}")

    print("\n--- DATA FORMAT (FEATURES) ---")
    for key in sample.keys():
        val = sample[key]
        val_type = type(val).__name__
        if isinstance(val, list):
            val_type = f"list (len={len(val)})"
        print(f"  - {key}: {val_type}")

    # Step 2: automatic schema extraction.
    print("\n" + "=" * 60)
    print("STEP 2: Automatic Schema Extraction")
    print("=" * 60)

    # Build schema cache from the full dataset.
    print("  Building schema cache (parsing all SQL queries)...")
    build_schema_cache(dataset)
    schema_info = extract_schema_from_sample(sample, dataset)

    print(f"\n  Database: {schema_info['db_id']}")
    print(f"  Table count: {len(schema_info['tables'])}")
    for table in schema_info["tables"]:
        print(f"    - {table['name']}: {table['columns']}")
    print(f"  Primary Keys: {schema_info['primary_keys']}")
    print(f"  Foreign Keys: {schema_info['foreign_keys']}")

    # Step 3: show three schema serialization formats.
    print("\n" + "=" * 60)
    print("STEP 3: Schema Serialization Formats")
    print("=" * 60)

    print("\n--- FORMAT A (Plain Text Listing) ---")
    schema_a = serialize_schema_format_a(schema_info)
    print(schema_a)

    print("\n--- FORMAT B (CREATE TABLE Syntax) ---")
    schema_b = serialize_schema_format_b(schema_info)
    print(schema_b)

    print("\n--- FORMAT C (Compact Notation) ---")
    schema_c = serialize_schema_format_c(schema_info)
    print(schema_c)

    # Step 4: prompt examples for baseline usage.
    print("\n" + "=" * 60)
    print("STEP 4: Prompt Examples")
    print("=" * 60)

    question = sample["question"]
    db_id = sample["db_id"]

    print("\n--- ZERO-SHOT PROMPT (Baseline 1) ---")
    zs_prompt = create_zero_shot_prompt(question, db_id, schema_a)
    print(zs_prompt)

    print("\n--- FEW-SHOT PROMPT (Baseline 2) ---")
    fs_prompt = create_few_shot_prompt(
        question, db_id, schema_a, DEFAULT_FEW_SHOT_EXAMPLES
    )
    print(fs_prompt[:500] + "...\n(truncated)")

    # Step 5: Mistral API connectivity test.
    print("\n" + "=" * 60)
    print("STEP 5: Mistral API Connection Test")
    print("=" * 60)

    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MISTRALAI_API_KEY")
    if not api_key:
        print("  [WARNING] MISTRAL_API_KEY not found. Check your .env file.")
        print("  Skipping API test...")
        return dataset

    client = get_mistral_client()

    # Simple API probe using a zero-shot prompt.
    print("\n  Sending a test prompt to Mistral API...")
    test_prompt = create_zero_shot_prompt(question, db_id, schema_a)
    prediction = get_sql_prediction(client, test_prompt, model_name="mistral-small-latest")
    target_sql = sample["query"]
    em = calculate_exact_match(prediction, target_sql)

    print(f"\n  Question : {question}")
    print(f"  Target SQL: {target_sql}")
    print(f"  Model SQL : {prediction}")
    print(f"  Exact Match: {'SUCCESS' if em == 1 else 'FAIL'}")

    # Persist dataset statistics for reporting.
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)

    stats = {
        "dataset_name": "Spider (xlangai/spider)",
        "train_size": len(dataset["train"]),
        "validation_size": len(dataset["validation"]),
        "sample_fields": list(sample.keys()),
        "unique_databases_train": len(set(s["db_id"] for s in dataset["train"])),
        "unique_databases_val": len(set(s["db_id"] for s in dataset["validation"])),
    }

    stats_path = os.path.join(data_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Dataset statistics saved: {stats_path}")

    # Prepare training data in JSONL format.
    create_instruction_dataset(dataset)

    return dataset

def create_instruction_dataset(dataset, output_path="data/train_formatted.jsonl"):
    """
    Prepare Spider training data in Alpaca format (JSONL) for LLM fine-tuning.
    """
    print("\n" + "=" * 60)
    print("Formatting Training Data (Instruction Tuning)")
    print("=" * 60)
    
    formatted_data = []
    
    # Only format the train split to avoid leaking targets from validation.
    print("  Converting training samples to Alpaca format...")
    for sample in dataset["train"]:
        # Use the shared schema extractor from utils.
        schema_info = extract_schema_from_sample(sample)
        # Use the CREATE TABLE style schema for clarity.
        schema_text = serialize_schema_format_b(schema_info)
        
        # Build an Alpaca-style JSON line.
        formatted_line = {
            "instruction": "You are an expert SQL developer. Your task is to translate the given natural language question into a valid executable SQL query.",
            "input": f"Database: {sample['db_id']}\nSchema:\n{schema_text}\n\nQuestion: {sample['question']}",
            "output": sample["query"]
        }
        formatted_data.append(formatted_line)
    
    # Write JSONL to disk.
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"  [OK] Total formatted training samples: {len(formatted_data)}")
    print(f"  [SAVED] Output path: {output_path}")


if __name__ == "__main__":
    main()
