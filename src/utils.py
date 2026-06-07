"""
utils.py — Core utility functions for Text-to-SQL project.

Contains:
  - Schema extraction from Spider dataset (Updated to use tables.json)
  - Schema serialization (3 formats: plain text, CREATE TABLE, compact)
  - Zero-shot and few-shot prompt builders
  - SQL normalization and exact match calculation
  - Mistral API wrapper for LLM inference
"""

import os
import re
import time
import json
from datasets import table
from dotenv import load_dotenv
from openai import OpenAI

# ─── Load environment variables ─────────────────────────────────────────────
load_dotenv()

# 1. SCHEMA EXTRACTION
#
# Schema metadata is loaded directly from Spider's official tables.json file.
# This ensures accurate primary-key and foreign-key relationships and avoids
# schema reconstruction error.
#

# Global schema cache built once and reused across evaluations.
_SCHEMA_CACHE = {}

def build_schema_cache(dataset=None, tables_json_path="data/tables.json"):
    global _SCHEMA_CACHE

    if _SCHEMA_CACHE:
        return _SCHEMA_CACHE

   # Verify that tables.json exists.
    if not os.path.exists(tables_json_path):
        raise FileNotFoundError(
            f"\n[CRITICAL ERROR] {tables_json_path} was not found.\n"
            "Please manually copy Spider's original tables.json file "
            "into the project's data directory."
        )

    # Load and parse Spider schema metadata.
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        primary_keys_idx = db.get("primary_keys", [])
        foreign_keys_idx = db.get("foreign_keys", [])
        column_types = db.get("column_types", [])

        tables_list = [{"name": t, "columns": []} for t in table_names]
        col_full_names = {}

        for idx, (tbl_idx, col_name) in enumerate(column_names):
            if tbl_idx == -1:
                continue 
            table_name = table_names[tbl_idx]
            col_type = column_types[idx] if idx < len(column_types) else "text"
            tables_list[tbl_idx]["columns"].append({
                "name": col_name,
                "type": col_type,
            })
            col_full_names[idx] = f"{table_name}.{col_name}"

        pk_list = [col_full_names[idx] for idx in primary_keys_idx if idx in col_full_names]
        fk_list = []
        for fk_idx, pk_idx in foreign_keys_idx:
            if fk_idx in col_full_names and pk_idx in col_full_names:
                fk_list.append((col_full_names[fk_idx], col_full_names[pk_idx]))

        _SCHEMA_CACHE[db_id] = {
            "db_id": db_id,
            "tables": tables_list,
            "primary_keys": pk_list,
            "foreign_keys": fk_list,
        }

    return _SCHEMA_CACHE


def extract_schema_from_sample(sample, dataset=None):
    """
    Retrieve schema information for a Spider sample using its db_id.

    The schema cache is initialized on first use by parsing tables.json.

    Args:
        sample: Spider dataset sample.
        dataset: Kept for backward compatibility.

    Returns:
        dict containing:
            - db_id
            - tables
            - primary_keys
            - foreign_keys
    """
    db_id = sample.get("db_id", "")

    if not _SCHEMA_CACHE:
        build_schema_cache()

    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]

    # Return an empty schema if db_id cannot be found.
    return {
        "db_id": db_id,
        "tables": [],
        "primary_keys": [],
        "foreign_keys": [],
    }



#  2. SCHEMA SERIALIZATION — 3 different formats for experimentation (plain text, SQL-like, compact)

def serialize_schema_format_a(schema_info):
    """
    Format A — Plain text listing.
    """
    lines = []
    for table in schema_info["tables"]:
        cols = ", ".join(col["name"] for col in table["columns"]) if table["columns"] else "(no columns)"
        lines.append(f"Table: {table['name']} | Columns: {cols}")

    if schema_info.get("primary_keys"):
        pk_str = ", ".join(schema_info["primary_keys"])
        lines.append(f"Primary Keys: {pk_str}")

    for fk, pk in schema_info.get("foreign_keys", []):
        lines.append(f"FK: {fk} -> {pk}")

    return "\n".join(lines)


def serialize_schema_format_b(schema_info):
    """
    Format B — SQL-style CREATE TABLE representation.
    """
    lines = []
    pk_set = set(schema_info.get("primary_keys", []))

    for table in schema_info["tables"]:
        col_defs = []
        for col in table["columns"]:
            col_name = col["name"]
            col_type = col["type"]
            full_name = f"{table['name']}.{col_name}"
            if full_name in pk_set:
                col_defs.append(f"  {col_name} {col_type} PRIMARY KEY")
            else:
                col_defs.append(f"  {col_name} {col_type}")

        cols_str = ",\n".join(col_defs) if col_defs else "  -- no columns"
        lines.append(f"CREATE TABLE {table['name']} (\n{cols_str}\n);")

    # Foreign key constraints
    for fk, pk in schema_info.get("foreign_keys", []):
        fk_table, fk_col = fk.split(".")
        pk_table, pk_col = pk.split(".")
        lines.append(
            f"-- FK: {fk_table}({fk_col}) REFERENCES {pk_table}({pk_col})"
        )

    return "\n".join(lines)


def serialize_schema_format_c(schema_info):
    """
    Format C — Compact token-efficient schema representation.
    """
    pk_set = set(schema_info.get("primary_keys", []))
    parts = []
    for table in schema_info["tables"]:
        col_parts = []
        for col in table["columns"]:
            col_name = col["name"]
            full_name = f"{table['name']}.{col_name}"

            if full_name in pk_set:
                col_parts.append(f"{col_name}[PK]")
            else:
                col_parts.append(col_name)
                cols_str = ", ".join(col_parts) if col_parts else ""
                parts.append(f"{table['name']}({cols_str})")

    result = " | ".join(parts)

    # FK info
    fk_parts = []
    for fk, pk in schema_info.get("foreign_keys", []):
        fk_parts.append(f"{fk}->{pk}")
    if fk_parts:
        result += " || FK: " + ", ".join(fk_parts)

    return result


SCHEMA_SERIALIZERS = {
    "format_a": serialize_schema_format_a,
    "format_b": serialize_schema_format_b,
    "format_c": serialize_schema_format_c,
}

#  3. PROMPT BUILDERS


def create_zero_shot_prompt(question, db_id, schema_text):
    """
    Create a zero-shot Text-to-SQL prompt.
    Args:
    - question (str): The natural language question to be translated into SQL.
    - db_id (str): The database identifier (used for context).
    - schema_text (str): The serialized schema information to include in the prompt.

    Returns:
    - str: The complete prompt to be sent to the LLM.

    """
    prompt = f"""You are an expert SQL developer. Your task is to translate the given natural language question into a valid executable SQL query.

Database: {db_id}
Schema:
{schema_text}

Question: {question}

Important: Return ONLY the SQL query, nothing else. Do not include explanations, markdown formatting, or code blocks."""
    return prompt


def create_few_shot_prompt(question, db_id, schema_text, examples):
    """
    Create a few-shot Text-to-SQL prompt (3-shot default).

    Args:
        question (str): Natural language question
        db_id (str): Database name
        schema_text (str): Serialized schema information
        examples (list): List of examples in [{question, sql}, ...] format

    Returns:
        str: Prompt to be sent to the LLM
    """
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"""
Example {i}:
Question: {ex['question']}
SQL: {ex['sql']}
"""

    prompt = f"""You are an expert SQL developer. Your task is to translate the given natural language question into a valid executable SQL query.

Database: {db_id}
Schema:
{schema_text}

Here are some examples of question-to-SQL translations:
{examples_text}
Now translate the following question:
Question: {question}

Important: Return ONLY the SQL query, nothing else. Do not include explanations, markdown formatting, or code blocks."""
    return prompt


# Default fallback few-shot examples.
DEFAULT_FEW_SHOT_EXAMPLES = [
    # 1. Simple aggregate with COUNT
    {
        "question": "How many records are in the table?",
        "sql": "SELECT COUNT(*) FROM table"
    },
    # 2. WHERE clause with multiple conditions and ORDER BY
    {
        "question": "List names and ages of people older than 20, ordered by age descending",
        "sql": "SELECT name, age FROM person WHERE age > 20 ORDER BY age DESC"
    },
    # 3. JOIN between two tables
    {
        "question": "Show names of people with their department names",
        "sql": "SELECT person.name, department.name FROM person JOIN department ON person.dept_id = department.id"
    },
    # 4. GROUP BY with aggregate function and HAVING
    {
        "question": "Find departments with more than 5 employees",
        "sql": "SELECT department, COUNT(*) as emp_count FROM employee GROUP BY department HAVING COUNT(*) > 5"
    },
    # 5. Subquery in WHERE clause
    {
        "question": "Find people whose age is above the average age",
        "sql": "SELECT name, age FROM person WHERE age > (SELECT AVG(age) FROM person)"
    },
    # 6. DISTINCT clause
    {
        "question": "What are all different countries in the dataset?",
        "sql": "SELECT DISTINCT country FROM person"
    },
    # 7. BETWEEN operator
    {
        "question": "Find records with values between 100 and 200",
        "sql": "SELECT id, value FROM record WHERE value BETWEEN 100 AND 200"
    },
    # 8. IN clause with multiple values
    {
        "question": "Show all employees from departments A, B, or C",
        "sql": "SELECT name, department FROM employee WHERE department IN ('A', 'B', 'C')"
    },
]

#  4. SQL NORMALIZATION & EXACT MATCH


def normalize_sql(sql):
    """
    Normalize SQL queries for exact-match evaluation

    Args:
        sql (str): Normalize sql query string

    Returns:
        str: Normalized SQL string
    """
    if not sql:
        return ""
    sql = sql.strip().lower()
    sql = sql.replace(";", "")
    # Remove Markdown code block formatting. 
    sql = re.sub(r"^```(sql)?", "", sql)
    sql = re.sub(r"```$", "", sql)
    sql = sql.strip()
    
    # Remove aliases introduced using AS.
    sql = re.sub(r"\s+as\s+\w+", "", sql)
    
    # Normalize whitespace inside parentheses.
    sql = re.sub(r"\(\s+", "(", sql)  # "( " → "("
    sql = re.sub(r"\s+\)", ")", sql)  # " )" → ")"
    
    # Normalize whitespace around commas.
    sql = re.sub(r"\s*,\s*", ", ", sql)
    
    # Normalize multiple whitespace characters into a single space.
    sql = re.sub(r"\s+", " ", sql)
    return sql


def calculate_exact_match(prediction, target):
    """
    Compute Exact Match (EM) between predicted and target SQL queries.

    Args:
        prediction (str): Model output SQL
        target (str): Target (gold) SQL

    Returns:
        int: 1 if they match, 0 otherwise
    """
    pred_norm = normalize_sql(prediction)
    target_norm = normalize_sql(target)
    return 1 if pred_norm == target_norm else 0



#  5. MISTRAL API WRAPPER

def get_mistral_client():
    """
    Creates a Mistral API client.

    Returns:
        OpenAI: Mistral-compatible OpenAI client
    """
    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MISTRALAI_API_KEY")
    base_url = os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_sql_prediction(client, prompt, model_name="mistral-small-latest", max_retries=3, retry_delay=2):
    """
    Generate a SQL prediction using the Mistral API.

    Automatically retries on transient API failures.

    Args:
        client: Mistral-compatible OpenAI client object
        prompt (str): Prompt to be sent to the LLM
        model_name (str): Name of the Mistral model to use (default: mistral-small-latest)
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retry attempts (seconds)

    Returns:
        str: Model output (SQL query)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=256,
            )
            result = response.choices[0].message.content.strip()
            # Remove any code block formatting if present.
            result = re.sub(r"^```(sql)?\n?", "", result)
            result = re.sub(r"\n?```$", "", result)
            return result.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [Retry {attempt+1}/{max_retries}] API error: {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                return f"API_ERROR: {e}"

#  6. HELPER FUNCTIONS

def get_few_shot_examples_from_dataset(dataset, db_id, n=3, exclude_idx=None):
    """
    Select few-shot examples from the same database (db_id).
    Returns default examples if not enough are found.

    Args:
        dataset: Hugging Face dataset object
        db_id (str): Database name
        n (int): Number of examples to select
        exclude_idx (int): Index of example to exclude

    Returns:
        list: [{question, sql}, ...] format examples
    """
    examples = []
    for i, sample in enumerate(dataset["train"]):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if sample["db_id"] == db_id:
            examples.append({
                "question": sample["question"],
                "sql": sample["query"]
            })
        if len(examples) >= n:
            break
    
    if len(examples) < n:
        remaining = n - len(examples)
        examples.extend(DEFAULT_FEW_SHOT_EXAMPLES[:remaining])

    return examples


def print_results_table(results):
    """
    Print results in a table format.

    Args:
        results (dict): {method_name: {em_score, total, correct, ...}}
    """
    print("\n" + "=" * 70)
    print(f"{'Method':<35} {'EM (%)':<10} {'Correct':<10} {'Total':<10}")
    print("=" * 70)
    for method, data in results.items():
        em_pct = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"{method:<35} {em_pct:<10.1f} {data['correct']:<10} {data['total']:<10}")
    print("=" * 70)
