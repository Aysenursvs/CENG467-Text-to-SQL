"""
evaluate_baseline.py — Baseline evaluation pipeline for Text-to-SQL.
Güncelleme: Execution Accuracy (EX) + Precision, Recall, F1 Metrikleri Eklendi!
"""

import os
import sys
import json
import time
import argparse
import sqlite3
from datetime import datetime

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    extract_schema_from_sample,
    SCHEMA_SERIALIZERS,
    create_zero_shot_prompt,
    create_few_shot_prompt,
    get_mistral_client,
    get_sql_prediction,
    calculate_exact_match,
    get_few_shot_examples_from_dataset,
    normalize_sql,
    print_results_table,
)

# --- AYARLAR ---
DB_DIR = "data/database"

def execute_sql(db_id, sql_query):
    """Verilen SQL sorgusunu SQLite veritabanında çalıştırır."""
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
        return f"SQL_HATA: {e}"

def calculate_set_metrics(gold_set, pred_set):
    """Hedef tablo ile modelin ürettiği tabloyu kesiştirerek IR metriklerini hesaplar."""
    # Eğer sorgulardan biri hata verdiyse ve küme (set) dönmediyse skorlar 0'dır
    if not isinstance(gold_set, set) or not isinstance(pred_set, set):
        return 0.0, 0.0, 0.0
    
    len_intersection = len(gold_set.intersection(pred_set))
    len_gold = len(gold_set)
    len_pred = len(pred_set)

    # İki sorgu da boş sonuç döndürdüyse (örn: tabloda gerçekten o veri yoksa) bu bir başarıdır
    if len_gold == 0 and len_pred == 0:
        return 1.0, 1.0, 1.0
    
    precision = len_intersection / len_pred if len_pred > 0 else 0.0
    recall = len_intersection / len_gold if len_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_baseline(
    dataset,
    client,
    model_name,
    baseline_type,
    schema_format,
    num_samples,
    delay_between_requests=1.5,
):
    serializer = SCHEMA_SERIALIZERS[schema_format]
    val_data = dataset["validation"]

    eval_samples = min(num_samples, len(val_data))

    em_correct = 0
    exec_correct = 0
    exec_total = 0
    
    # Yeni metrikler için toplam sayaçlar
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    
    predictions = []
    errors = []

    print(f"\n{'='*70}")
    print(f"Baseline: {baseline_type.upper()} | Schema: {schema_format}")
    print(f"Örnekler: {eval_samples}")
    print(f"{'='*70}")

    for i in range(eval_samples):
        sample = val_data[i]
        question = sample["question"]
        db_id = sample["db_id"]
        target_sql = sample["query"]

        schema_info = extract_schema_from_sample(sample)
        schema_text = serializer(schema_info)

        if baseline_type == "zero_shot":
            prompt = create_zero_shot_prompt(question, db_id, schema_text)
        elif baseline_type == "few_shot":
            examples = get_few_shot_examples_from_dataset(dataset, db_id, n=3)
            prompt = create_few_shot_prompt(question, db_id, schema_text, examples)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        prediction = get_sql_prediction(client, prompt, model_name=model_name)

        # 1. Exact Match
        em = calculate_exact_match(prediction, target_sql)
        em_correct += em

        # 2. Execution & Set Metrics
        gold_result = execute_sql(db_id, target_sql)
        pred_result = execute_sql(db_id, prediction)
        
        ex_match = 0
        p, r, f1 = calculate_set_metrics(gold_result, pred_result)
        
        sum_precision += p
        sum_recall += r
        sum_f1 += f1
        
        if "HATA" not in str(gold_result) and "HATA" not in str(pred_result) and gold_result != "DB_NOT_FOUND" and pred_result != "DB_NOT_FOUND":
            exec_total += 1
            if gold_result == pred_result:
                exec_correct += 1
                ex_match = 1

        pred_record = {
            "index": i,
            "db_id": db_id,
            "question": question,
            "target_sql": target_sql,
            "predicted_sql": prediction,
            "exact_match": em,
            "exec_match": ex_match,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4)
        }
        predictions.append(pred_record)

        if em == 0 or ex_match == 0:
            errors.append(pred_record)

        em_cumulative = (em_correct / (i+1)) * 100
        ex_cumulative = (exec_correct / exec_total) * 100 if exec_total > 0 else 0
        
        status = "✅" if ex_match == 1 else "❌"
        print(f"  [{i+1:3d}/{eval_samples}] {status} EM={em_cumulative:.1f}% | EX={ex_cumulative:.1f}% | F1={f1:.2f} | DB: {db_id}")

        if i < eval_samples - 1:
            time.sleep(delay_between_requests)

    em_score = (em_correct / eval_samples) * 100 if eval_samples > 0 else 0
    ex_score = (exec_correct / exec_total) * 100 if exec_total > 0 else 0
    
    # Ortalama IR Metrikleri
    avg_precision = (sum_precision / eval_samples) * 100
    avg_recall = (sum_recall / eval_samples) * 100
    avg_f1 = (sum_f1 / eval_samples) * 100

    print(f"\n--- SONUÇ: {baseline_type.upper()} ({schema_format}) ---")
    print(f"  Exact Match: {em_correct}/{eval_samples} = {em_score:.2f}%")
    print(f"  Execution Accuracy: {exec_correct}/{exec_total} = {ex_score:.2f}%")
    print(f"  Avg Precision: {avg_precision:.2f}%")
    print(f"  Avg Recall:    {avg_recall:.2f}%")
    print(f"  Avg F1-Score:  {avg_f1:.2f}%")

    return {
        "baseline_type": baseline_type,
        "schema_format": schema_format,
        "model_name": model_name,
        "num_samples": eval_samples,
        "em_correct": em_correct,
        "exec_correct": exec_correct,
        "exec_total": exec_total,
        "em_score": em_score,
        "ex_score": ex_score,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "predictions": predictions,
        "error_examples": errors[:10],
        "timestamp": datetime.now().isoformat(),
    }


def run_full_evaluation(args):
    print("=" * 70)
    print("SPIDER VALIDATION SET — BASELINE EVALUATION (WITH EXEC & F1 METRICS)")
    print("=" * 70)

    print("\nSpider veriseti yükleniyor...")
    dataset = load_dataset("xlangai/spider")
    
    print(f"\nMistral API ayarlandı: {args.model}...")
    client = get_mistral_client()

    all_results = {}
    summary = {}

    print("\n\n" + "#" * 70)
    print("# BASELINE 1: ZERO-SHOT PROMPTING")
    print("#" * 70)

    zs_results = evaluate_baseline(
        dataset, client, args.model, "zero_shot", args.schema_format,
        args.num_samples, args.delay
    )
    all_results["zero_shot"] = zs_results

    print("\n\n" + "#" * 70)
    print("# BASELINE 2: FEW-SHOT PROMPTING (3-shot)")
    print("#" * 70)

    fs_results = evaluate_baseline(
        dataset, client, args.model, "few_shot", args.schema_format,
        args.num_samples, args.delay
    )
    all_results["few_shot"] = fs_results

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "baseline_results_with_f1.json")
    save_data = {
        "experiment_config": {
            "model": args.model,
            "schema_format": args.schema_format,
            "num_samples": args.num_samples,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {
            "zero_shot": {
                "em_score": zs_results["em_score"],
                "ex_score": zs_results["ex_score"],
                "precision": zs_results["avg_precision"],
                "recall": zs_results["avg_recall"],
                "f1_score": zs_results["avg_f1"],
            },
            "few_shot": {
                "em_score": fs_results["em_score"],
                "ex_score": fs_results["ex_score"],
                "precision": fs_results["avg_precision"],
                "recall": fs_results["avg_recall"],
                "f1_score": fs_results["avg_f1"],
            },
        }
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("🏆 FİNAL BASELINE RAPORU (ZENGİNLEŞTİRİLMİŞ METRİKLER)")
    print("=" * 70)
    print(f"ZERO-SHOT -> EM: %{zs_results['em_score']:.2f} | EX: %{zs_results['ex_score']:.2f} | P: %{zs_results['avg_precision']:.2f} | R: %{zs_results['avg_recall']:.2f} | F1: %{zs_results['avg_f1']:.2f}")
    print(f"FEW-SHOT  -> EM: %{fs_results['em_score']:.2f} | EX: %{fs_results['ex_score']:.2f} | P: %{fs_results['avg_precision']:.2f} | R: %{fs_results['avg_recall']:.2f} | F1: %{fs_results['avg_f1']:.2f}")
    print("=" * 70)
    print(f"Detaylı sonuçlar JSON formatında kaydedildi: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL Baseline Evaluation on Spider")
    parser.add_argument("--num_samples", type=int, default=100, help="Değerlendirilecek örnek sayısı")
    parser.add_argument("--schema_format", type=str, default="format_b", choices=["format_a", "format_b", "format_c"])
    parser.add_argument("--model", type=str, default="mistral-small-latest")
    parser.add_argument("--delay", type=float, default=1.5)

    args = parser.parse_args()
    run_full_evaluation(args)


if __name__ == "__main__":
    main()