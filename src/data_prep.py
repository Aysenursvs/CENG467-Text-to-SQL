"""
data_prep.py — Spider verisetini indirme, inceleme ve ön işleme scripti.

Bu script:
  1. Spider verisetini Hugging Face'den indirir
  2. Verisetinin yapısını ve boyutlarını ekrana basar
  3. Şema bilgisini otomatik olarak çıkarır
  4. 3 farklı şema serileştirme formatını gösterir
  5. Zero-shot ve few-shot prompt örnekleri oluşturur
    6. OpenRouter API bağlantısını test eder
"""

import os
import sys
import json

from datasets import load_dataset

# src/ klasöründeki utils modülünü import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    extract_schema_from_sample,
    build_schema_cache,
    serialize_schema_format_a,
    serialize_schema_format_b,
    serialize_schema_format_c,
    create_zero_shot_prompt,
    create_few_shot_prompt,
    get_openrouter_client,
    get_sql_prediction,
    calculate_exact_match,
    DEFAULT_FEW_SHOT_EXAMPLES,
)


def main():
    # ─── 1. Spider verisetini indir ──────────────────────────────────────────
    print("=" * 60)
    print("ADIM 1: Spider Veriseti İndiriliyor / Yükleniyor...")
    print("=" * 60)
    dataset = load_dataset("xlangai/spider")

    # ─── 2. Verisetinin boyutları ────────────────────────────────────────────
    print("\n--- VERİSETİ BOYUTLARI ---")
    print(f"  Eğitim (Train) seti   : {len(dataset['train'])} örnek")
    print(f"  Doğrulama (Validation): {len(dataset['validation'])} örnek")

    # ─── 3. İlk örneği incele ────────────────────────────────────────────────
    sample = dataset["train"][0]

    print("\n--- İLK ÖRNEK İNCELEMESİ ---")
    print(f"  Veritabanı Adı (db_id) : {sample['db_id']}")
    print(f"  Doğal Dil Sorusu       : {sample['question']}")
    print(f"  Hedef SQL Sorgusu      : {sample['query']}")

    print("\n--- VERİ FORMATI (FEATURES) ---")
    for key in sample.keys():
        val = sample[key]
        val_type = type(val).__name__
        if isinstance(val, list):
            val_type = f"list (len={len(val)})"
        print(f"  - {key}: {val_type}")

    # ─── 4. Otomatik şema çıkarma ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ADIM 2: Otomatik Şema Çıkarma (Schema Extraction)")
    print("=" * 60)

    # Tüm veri setinden şema cache'i oluştur
    print("  Schema cache oluşturuluyor (tüm SQL sorguları parse ediliyor)...")
    build_schema_cache(dataset)
    schema_info = extract_schema_from_sample(sample, dataset)

    print(f"\n  Veritabanı: {schema_info['db_id']}")
    print(f"  Tablo sayısı: {len(schema_info['tables'])}")
    for table in schema_info["tables"]:
        print(f"    - {table['name']}: {table['columns']}")
    print(f"  Primary Keys: {schema_info['primary_keys']}")
    print(f"  Foreign Keys: {schema_info['foreign_keys']}")

    # ─── 5. 3 farklı şema serileştirme formatı ──────────────────────────────
    print("\n" + "=" * 60)
    print("ADIM 3: Şema Serileştirme Formatları")
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

    # ─── 6. Prompt örnekleri ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ADIM 4: Prompt Örnekleri")
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
    print(fs_prompt[:500] + "...\n(kısaltıldı)")

    # ─── 7. OpenRouter API testi ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ADIM 5: OpenRouter API Bağlantı Testi")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  [UYARI] OPENROUTER_API_KEY bulunamadı! .env dosyanı kontrol et.")
        print("  API testini atlıyorum...")
        return dataset

    client = get_openrouter_client()

    # Basit test
    print("\n  OpenRouter API'ye test sorgusu gönderiliyor...")
    test_prompt = create_zero_shot_prompt(question, db_id, schema_a)
    prediction = get_sql_prediction(client, test_prompt, model_name="inclusionai/ring-2.6-1t:free")
    target_sql = sample["query"]
    em = calculate_exact_match(prediction, target_sql)

    print(f"\n  Soru     : {question}")
    print(f"  Hedef SQL: {target_sql}")
    print(f"  Model SQL: {prediction}")
    print(f"  Exact Match: {'✅ BAŞARILI' if em == 1 else '❌ BAŞARISIZ'}")

    # ─── 8. Veri seti istatistiklerini kaydet ────────────────────────────────
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
    print(f"\n  Veri seti istatistikleri kaydedildi: {stats_path}")

    return dataset


if __name__ == "__main__":
    main()
