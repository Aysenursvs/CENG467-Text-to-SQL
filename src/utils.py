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
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ─── Load environment variables ─────────────────────────────────────────────
load_dotenv()



# =============================================================================
#  1. SCHEMA EXTRACTION — Orijinal tables.json'dan şema bilgisi çıkarma
# =============================================================================
#
# NOT: Artık SQL'leri regex ile parse etmek (tahmin etmek) yerine Spider'ın 
# resmi tables.json dosyasını kullanıyoruz. Bu sayede Primary Key ve Foreign
# Key ilişkilerini modele %100 doğru aktararak halüsinasyonları çözeceğiz.
#

# Global schema cache — bir kez oluşturulur, sonra tekrar kullanılır
_SCHEMA_CACHE = {}


def build_schema_cache(dataset=None, tables_json_path="data/tables.json"):
    """
    Spider'ın orijinal tables.json dosyasını okuyarak her db_id için
    kusursuz bir şema sözlüğü oluşturur (PK ve FK dahil).
    Eğer dosya yoksa Spider'ın resmi GitHub reposundan otomatik indirir.

    Args:
        dataset: (Geriye dönük uyumluluk için bırakıldı)
        tables_json_path: tables.json dosyasının kaydedileceği/okunacağı yol

    Returns:
        dict: {db_id: {"tables": [{"name": str, "columns": [str]}], ...}}
    """
    global _SCHEMA_CACHE

    if _SCHEMA_CACHE:
        return _SCHEMA_CACHE

    # Dosya yoksa internetten indir
    if not os.path.exists(tables_json_path):
        os.makedirs(os.path.dirname(tables_json_path) or ".", exist_ok=True)
        print(f"  [Schema Extractor] {tables_json_path} bulunamadı.")
        print(f"  [Schema Extractor] Orijinal tables.json GitHub'dan indiriliyor...")
        url = "https://raw.githubusercontent.com/taoyds/spider/master/tables.json"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(tables_json_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("  [Schema Extractor] İndirme başarılı!")
        except Exception as e:
            raise RuntimeError(f"tables.json indirilemedi: {e}")

    # JSON dosyasını oku ve parse et
    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        primary_keys_idx = db.get("primary_keys", [])
        foreign_keys_idx = db.get("foreign_keys", [])

        # 1. Tablo iskeletlerini oluştur
        tables_list = [{"name": t, "columns": []} for t in table_names]

        # 2. Kolonları tablolara yerleştir
        # Spider veri formatında column_names şöyledir: [[tablo_indexi, "kolon_adi"], ...]
        # Index 0 genellikle [-1, "*"] olur (tüm tablolar için joker kolon), bunu atlıyoruz.
        col_full_names = {}  # index -> "tablo_adi.kolon_adi"

        for idx, (tbl_idx, col_name) in enumerate(column_names):
            if tbl_idx == -1:
                continue  # "*" kolonunu yoksay
            
            table_name = table_names[tbl_idx]
            tables_list[tbl_idx]["columns"].append(col_name)
            col_full_names[idx] = f"{table_name}.{col_name}"

        # 3. Primary Key'leri metne çevir (Örn: "department.Department_ID")
        pk_list = [col_full_names[idx] for idx in primary_keys_idx if idx in col_full_names]

        # 4. Foreign Key'leri metne çevir (Örn: ["management.head_ID", "head.head_ID"])
        fk_list = []
        for fk_idx, pk_idx in foreign_keys_idx:
            if fk_idx in col_full_names and pk_idx in col_full_names:
                fk_list.append((col_full_names[fk_idx], col_full_names[pk_idx]))

        # Cache'e kaydet
        _SCHEMA_CACHE[db_id] = {
            "db_id": db_id,
            "tables": tables_list,
            "primary_keys": pk_list,
            "foreign_keys": fk_list,
        }

    return _SCHEMA_CACHE


def extract_schema_from_sample(sample, dataset=None):
    """
    Bir Spider örneğinin db_id'sine göre kusursuz şema bilgisini döndürür.
    İlk çağrıda tables.json parse edilerek cache oluşturulur.

    Args:
        sample: Spider verisetinden bir örnek
        dataset: (Geriye dönük uyumluluk için bırakıldı)

    Returns:
        dict: {"db_id", "tables", "primary_keys", "foreign_keys"}
    """
    db_id = sample.get("db_id", "")

    if not _SCHEMA_CACHE:
        build_schema_cache()

    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]

    # Eğer db_id bir şekilde tables.json'da yoksa (ki Spider'da hepsi vardır)
    # Hata fırlatmak yerine boş bir taslak döndür (kodu çökertmemek için)
    return {
        "db_id": db_id,
        "tables": [],
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


# =============================================================================
#  4. SQL NORMALIZATION & EXACT MATCH
# =============================================================================

def normalize_sql(sql):
    """
    SQL sorgusunu normalize eder: küçük harf, fazla boşluk temizleme,
    noktalı virgül kaldırma, AS clause'ları kaldırma.

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
    
    # AS clause'ları ve alias'ları kaldır
    # Örneğin: "SELECT COUNT(*) AS total_singers" → "SELECT COUNT(*)"
    sql = re.sub(r"\s+as\s+\w+", "", sql)
    
    # Parantez İÇİndeki boşlukları kaldır (ama dış boşlukları koru)
    sql = re.sub(r"\(\s+", "(", sql)  # "( " → "("
    sql = re.sub(r"\s+\)", ")", sql)  # " )" → ")"
    
    # Virgül çevresindeki boşlukları normalize et
    sql = re.sub(r"\s*,\s*", ", ", sql)
    
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
#  5. MISTRAL API WRAPPER
# =============================================================================

def get_mistral_client():
    """
    Mistral API istemcisi oluşturur.

    Returns:
        OpenAI: Mistral uyumlu OpenAI client
    """
    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("MISTRALAI_API_KEY")
    base_url = os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_sql_prediction(client, prompt, model_name="mistral-small-latest", max_retries=3, retry_delay=2):
    """
    Mistral API ile SQL tahmini alır. Rate limit hatalarında retry yapar.

    Args:
        client: Mistral uyumlu istemci nesnesi
        prompt (str): LLM'e gönderilecek prompt
            model_name (str): Kullanılacak Mistral model adı (varsayılan: mistral-small-latest)
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