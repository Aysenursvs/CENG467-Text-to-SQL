"""
utils.py — Core utility functions for Text-to-SQL project.

Contains:
  - Schema extraction from Spider dataset
  - Schema serialization (3 formats: plain text, CREATE TABLE, compact)
  - Zero-shot and few-shot prompt builders
  - SQL normalization and exact match calculation
    - OpenRouter API wrapper for LLM inference
"""

import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

# ─── Load environment variables ─────────────────────────────────────────────
load_dotenv()



# =============================================================================
#  1. SCHEMA EXTRACTION — SQL sorgularından şema bilgisi çıkarma
# =============================================================================
#
# NOT: HuggingFace'deki Spider verisetinde (xlangai/spider) tablo/kolon
# isimleri doğrudan verilmiyor. Bu yüzden eğitim setindeki SQL sorgularını
# parse ederek her db_id için bir şema sözlüğü oluşturuyoruz.
#

import sqlparse

# Global schema cache — bir kez oluşturulur, sonra tekrar kullanılır
_SCHEMA_CACHE = {}


def _parse_tables_and_columns_from_sql(sql):
    """
    Bir SQL sorgusundan tablo ve kolon isimlerini çıkarır.

    Basit bir regex/keyword-based parser kullanır.
    Mükemmel değildir ama Spider verisetindeki sorguların çoğu için yeterlidir.

    Returns:
        tables (set): Bulunan tablo isimleri
        columns (dict): {tablo_adı: set(kolon_adları)} — eşleşebilenler
        standalone_columns (set): Tabloya eşleştirilemeyen kolon isimleri
    """
    sql_upper = sql.upper()
    sql_clean = sql.strip().rstrip(";")

    tables = set()
    columns = {}
    standalone_columns = set()

    # SQL anahtar kelimeleri (bunları tablo/kolon adı olarak alma)
    sql_keywords = {
        "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER",
        "ON", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE", "IS",
        "NULL", "AS", "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "UNION",
        "ALL", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX", "ASC", "DESC",
        "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE",
        "TABLE", "DROP", "ALTER", "INDEX", "VIEW", "CASE", "WHEN", "THEN",
        "ELSE", "END", "CROSS", "NATURAL", "USING", "EXCEPT", "INTERSECT",
        "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "INT", "TEXT", "FLOAT",
        "REAL", "INTEGER", "VARCHAR", "CHAR", "BOOLEAN", "DATE", "DATETIME",
        "VALUE", "TRUE", "FALSE", "WITH", "RECURSIVE", "OVER", "PARTITION",
        "ROWS", "RANGE", "PRECEDING", "FOLLOWING", "CURRENT", "ROW", "OFFSET",
    }

    # 1. FROM ve JOIN'den tablo isimlerini çıkar
    # FROM tablo1, tablo2  veya  FROM tablo1 JOIN tablo2
    tokens = re.split(r"[\s,()]+", sql_clean)
    i = 0
    while i < len(tokens):
        token_upper = tokens[i].upper()
        if token_upper in ("FROM", "JOIN"):
            # Sonraki token tablo adı olmalı
            if i + 1 < len(tokens):
                candidate = tokens[i + 1].strip("`\"'[]")
                if candidate.upper() not in sql_keywords and candidate:
                    tables.add(candidate.lower())
        i += 1

    # 2. "tablo.kolon" formatındaki referansları çıkar
    dot_pattern = re.findall(r"(\w+)\.(\w+)", sql_clean)
    for tbl, col in dot_pattern:
        tbl_lower = tbl.lower()
        col_lower = col.lower()
        if tbl_lower not in {k.lower() for k in sql_keywords}:
            tables.add(tbl_lower)
            if tbl_lower not in columns:
                columns[tbl_lower] = set()
            if col_lower not in {k.lower() for k in sql_keywords}:
                columns[tbl_lower].add(col_lower)

    # 3. SELECT ve WHERE'den kolon isimlerini çıkar (tablo.kolon olmayanlar)
    select_where_pattern = re.findall(
        r"(?:SELECT|WHERE|ON|BY|HAVING)\s+(.+?)(?:\s+FROM|\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|\s+LIMIT|$)",
        sql_clean, re.IGNORECASE
    )
    for clause in select_where_pattern:
        col_tokens = re.findall(r"\b(\w+)\b", clause)
        for ct in col_tokens:
            if ct.upper() not in sql_keywords and not ct.isdigit() and len(ct) > 1:
                standalone_columns.add(ct.lower())

    return tables, columns, standalone_columns


def build_schema_cache(dataset):
    """
    Tüm eğitim verisindeki SQL sorgularını parse ederek her db_id için
    bir şema sözlüğü oluşturur.

    Args:
        dataset: Hugging Face dataset nesnesi (train + validation)

    Returns:
        dict: {db_id: {"tables": [{"name": str, "columns": [str]}], ...}}
    """
    global _SCHEMA_CACHE

    if _SCHEMA_CACHE:
        return _SCHEMA_CACHE

    db_schemas = {}  # {db_id: {table_name: set(columns)}}

    # Train ve validation'daki tüm sorguları tara
    for split_name in ["train", "validation"]:
        if split_name not in dataset:
            continue
        for sample in dataset[split_name]:
            db_id = sample["db_id"]
            sql = sample["query"]

            if db_id not in db_schemas:
                db_schemas[db_id] = {}

            tables, columns, standalone_cols = _parse_tables_and_columns_from_sql(sql)

            # Tabloları ekle
            for tbl in tables:
                if tbl not in db_schemas[db_id]:
                    db_schemas[db_id][tbl] = set()

            # Kolon eşleştirmelerini ekle
            for tbl, cols in columns.items():
                if tbl not in db_schemas[db_id]:
                    db_schemas[db_id][tbl] = set()
                db_schemas[db_id][tbl].update(cols)

            # Standalone kolonları mevcut tablolara ata (tek tablo varsa)
            if len(tables) == 1 and standalone_cols:
                tbl = list(tables)[0]
                db_schemas[db_id][tbl].update(standalone_cols)

    # Cache formatına dönüştür
    for db_id, tables_dict in db_schemas.items():
        _SCHEMA_CACHE[db_id] = {
            "db_id": db_id,
            "tables": [
                {"name": tbl, "columns": sorted(list(cols))}
                for tbl, cols in sorted(tables_dict.items())
            ],
            "primary_keys": [],
            "foreign_keys": [],
        }

    return _SCHEMA_CACHE


def extract_schema_from_sample(sample, dataset=None):
    """
    Bir Spider örneğinin db_id'sine göre şema bilgisini döndürür.
    İlk çağrıda tüm veri setinden şema cache'i oluşturulur.

    Args:
        sample: Spider verisetinden bir örnek
        dataset: (opsiyonel) Cache oluşturmak için veri seti

    Returns:
        dict: {"db_id", "tables", "primary_keys", "foreign_keys"}
    """
    db_id = sample.get("db_id", "")

    if dataset is not None and not _SCHEMA_CACHE:
        build_schema_cache(dataset)

    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]

    # Cache'de yoksa, tek sorgudan çıkar
    tables, columns, _ = _parse_tables_and_columns_from_sql(sample.get("query", ""))
    table_list = []
    for tbl in sorted(tables):
        cols = sorted(list(columns.get(tbl, set())))
        table_list.append({"name": tbl, "columns": cols})

    return {
        "db_id": db_id,
        "tables": table_list,
        "primary_keys": [],
        "foreign_keys": [],
    }


# =============================================================================
#  2. SCHEMA SERIALIZATION — 3 farklı format
# =============================================================================

def serialize_schema_format_a(schema_info):
    """
    Format A — Plain text listing (okunabilir format).
    Örnek:
        Table: students | Columns: id, name, age, gpa
        FK: enrollments.student_id -> students.id
    """
    lines = []
    for table in schema_info["tables"]:
        cols = ", ".join(table["columns"]) if table["columns"] else "(no columns)"
        lines.append(f"Table: {table['name']} | Columns: {cols}")

    if schema_info.get("primary_keys"):
        pk_str = ", ".join(schema_info["primary_keys"])
        lines.append(f"Primary Keys: {pk_str}")

    for fk, pk in schema_info.get("foreign_keys", []):
        lines.append(f"FK: {fk} -> {pk}")

    return "\n".join(lines)


def serialize_schema_format_b(schema_info):
    """
    Format B — CREATE TABLE syntax (SQL benzeri format).
    Örnek:
        CREATE TABLE students (
          id PRIMARY KEY, name, age, gpa
        );
    """
    lines = []
    pk_set = set(schema_info.get("primary_keys", []))

    for table in schema_info["tables"]:
        col_defs = []
        for col in table["columns"]:
            full_name = f"{table['name']}.{col}"
            if full_name in pk_set:
                col_defs.append(f"  {col} PRIMARY KEY")
            else:
                col_defs.append(f"  {col}")

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
    Format C — Compact notation (kısa/token-efficient format).
    Örnek:
        students(id[PK], name, age, gpa) | courses(id[PK], title, credits)
    """
    pk_set = set(schema_info.get("primary_keys", []))
    parts = []
    for table in schema_info["tables"]:
        col_parts = []
        for col in table["columns"]:
            full_name = f"{table['name']}.{col}"
            if full_name in pk_set:
                col_parts.append(f"{col}[PK]")
            else:
                col_parts.append(col)
        cols_str = ", ".join(col_parts) if col_parts else ""
        parts.append(f"{table['name']}({cols_str})")

    result = " | ".join(parts)

    # FK bilgisi
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


# =============================================================================
#  3. PROMPT BUILDERS — Zero-shot ve Few-shot
# =============================================================================

def create_zero_shot_prompt(question, db_id, schema_text):
    """
    Zero-shot prompt oluşturur.

    Args:
        question (str): Doğal dil sorusu
        db_id (str): Veritabanı adı
        schema_text (str): Serileştirilmiş şema bilgisi

    Returns:
        str: LLM'e gönderilecek prompt
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
    Few-shot prompt oluşturur (3-shot default).

    Args:
        question (str): Doğal dil sorusu
        db_id (str): Veritabanı adı
        schema_text (str): Serileştirilmiş şema bilgisi
        examples (list): [{question, sql}, ...] formatında örnekler

    Returns:
        str: LLM'e gönderilecek prompt
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


# Varsayılan few-shot örnekleri (Spider verisetinden seçilmiş basit örnekler)
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "question": "How many departments are there?",
        "sql": "SELECT COUNT(*) FROM department"
    },
    {
        "question": "What are the names of all students older than 20?",
        "sql": "SELECT name FROM students WHERE age > 20"
    },
    {
        "question": "Show the name and budget of departments with more than 100 employees, ordered by budget.",
        "sql": "SELECT name, budget FROM department WHERE num_employees > 100 ORDER BY budget DESC"
    },
]


# =============================================================================
#  4. SQL NORMALIZATION & EXACT MATCH
# =============================================================================

def normalize_sql(sql):
    """
    SQL sorgusunu normalize eder: küçük harf, fazla boşluk temizleme,
    noktalı virgül kaldırma.

    Args:
        sql (str): Normalize edilecek SQL sorgusu

    Returns:
        str: Normalize edilmiş SQL
    """
    if not sql:
        return ""
    sql = sql.strip().lower()
    sql = sql.replace(";", "")
    # Markdown code block temizliği
    sql = re.sub(r"^```(sql)?", "", sql)
    sql = re.sub(r"```$", "", sql)
    sql = sql.strip()
    # Fazla boşlukları tek boşluğa indir
    sql = re.sub(r"\s+", " ", sql)
    return sql


def calculate_exact_match(prediction, target):
    """
    Exact Match (EM) skoru hesaplar.

    Args:
        prediction (str): Model çıktısı SQL
        target (str): Hedef (gold) SQL

    Returns:
        int: 1 eşleşiyorsa, 0 eşleşmiyorsa
    """
    pred_norm = normalize_sql(prediction)
    target_norm = normalize_sql(target)
    return 1 if pred_norm == target_norm else 0


# =============================================================================
#  5. OPENROUTER API WRAPPER
# =============================================================================

def get_openrouter_client():
    """
    OpenRouter API istemcisi oluşturur.

    Returns:
        OpenAI: OpenRouter uyumlu istemci
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    extra_headers = {}
    http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    app_name = os.environ.get("OPENROUTER_APP_NAME")
    if http_referer:
        extra_headers["HTTP-Referer"] = http_referer
    if app_name:
        extra_headers["X-Title"] = app_name

    if extra_headers:
        return OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)
    return OpenAI(api_key=api_key, base_url=base_url)


def get_sql_prediction(client, prompt, model_name="inclusionai/ring-2.6-1t:free", max_retries=3, retry_delay=2):
    """
    OpenRouter API ile SQL tahmini alır. Rate limit hatalarında retry yapar.

    Args:
        client: OpenRouter uyumlu istemci nesnesi
        prompt (str): LLM'e gönderilecek prompt
            model_name (str): Kullanılacak OpenRouter model adı (varsayılan: inclusionai/ring-2.6-1t:free)
        max_retries (int): Maksimum deneme sayısı
        retry_delay (int): Denemeler arası bekleme süresi (saniye)

    Returns:
        str: Model çıktısı (SQL sorgusu)
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
            # Markdown code block temizliği
            result = re.sub(r"^```(sql)?\n?", "", result)
            result = re.sub(r"\n?```$", "", result)
            return result.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [Retry {attempt+1}/{max_retries}] API hatası: {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                return f"API_ERROR: {e}"


# =============================================================================
#  6. HELPER FUNCTIONS
# =============================================================================

def get_few_shot_examples_from_dataset(dataset, db_id, n=3, exclude_idx=None):
    """
    Aynı veritabanından (db_id) few-shot örnekleri seçer.
    Bulamazsa varsayılan örnekleri döndürür.

    Args:
        dataset: Hugging Face dataset nesnesi
        db_id (str): Veritabanı adı
        n (int): Seçilecek örnek sayısı
        exclude_idx (int): Hariç tutulacak örnek indeksi

    Returns:
        list: [{question, sql}, ...] formatında örnekler
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

    # Yeterince örnek bulunamazsa varsayılanları kullan
    if len(examples) < n:
        remaining = n - len(examples)
        examples.extend(DEFAULT_FEW_SHOT_EXAMPLES[:remaining])

    return examples


def print_results_table(results):
    """
    Sonuçları tablo formatında yazdırır.

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
