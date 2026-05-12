"""
evaluate.py — Baseline evaluation pipeline for Text-to-SQL.

This script evaluates two baseline prompting strategies on the Spider
validation set using the OpenRouter API:
  - Baseline 1: Zero-shot prompting
  - Baseline 2: Few-shot prompting (3-shot)

Metrics: Exact Match (EM)

Usage:
    python src/evaluate.py --num_samples 50 --schema_format format_a
    python src/evaluate.py --num_samples 100 --schema_format format_b
    python src/evaluate.py --num_samples 50 --schema_format format_b --delay 4.0
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    extract_schema_from_sample,
    SCHEMA_SERIALIZERS,
    create_zero_shot_prompt,
    create_few_shot_prompt,
    get_openrouter_client,
    get_sql_prediction,
    calculate_exact_match,
    get_few_shot_examples_from_dataset,
    normalize_sql,
    print_results_table,
)


def evaluate_baseline(
    dataset,
    client,
    model_name,
    baseline_type,
    schema_format,
    num_samples,
    delay_between_requests=1.5,
):
    """
    Bir baseline stratejisini Spider validation set üzerinde değerlendirir.

    Args:
        dataset: HuggingFace dataset nesnesi
        client: OpenRouter istemci nesnesi
        model_name (str): OpenRouter model adı
        baseline_type (str): "zero_shot" veya "few_shot"
        schema_format (str): "format_a", "format_b", "format_c"
        num_samples (int): Değerlendirilecek örnek sayısı
        delay_between_requests (float): API istekleri arası bekleme süresi (saniye)

    Returns:
        dict: Evaluation sonuçları
    """
    serializer = SCHEMA_SERIALIZERS[schema_format]
    val_data = dataset["validation"]

    # Değerlendirilecek örnek sayısını sınırla
    eval_samples = min(num_samples, len(val_data))

    correct = 0
    total = 0
    predictions = []
    errors = []

    print(f"\n{'='*60}")
    print(f"Baseline: {baseline_type.upper()} | Schema: {schema_format}")
    print(f"Örnekler: {eval_samples}")
    print(f"{'='*60}")

    for i in range(eval_samples):
        sample = val_data[i]
        question = sample["question"]
        db_id = sample["db_id"]
        target_sql = sample["query"]

        # Schema bilgisini çıkar ve serileştir
        schema_info = extract_schema_from_sample(sample)
        schema_text = serializer(schema_info)

        # Prompt oluştur
        if baseline_type == "zero_shot":
            prompt = create_zero_shot_prompt(question, db_id, schema_text)
        elif baseline_type == "few_shot":
            examples = get_few_shot_examples_from_dataset(dataset, db_id, n=3)
            prompt = create_few_shot_prompt(question, db_id, schema_text, examples)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        # Model çıktısını al
        prediction = get_sql_prediction(client, prompt, model_name=model_name)

        # Exact match hesapla
        em = calculate_exact_match(prediction, target_sql)
        correct += em
        total += 1

        # Sonucu kaydet
        pred_record = {
            "index": i,
            "db_id": db_id,
            "question": question,
            "target_sql": target_sql,
            "predicted_sql": prediction,
            "exact_match": em,
        }
        predictions.append(pred_record)

        if em == 0:
            errors.append(pred_record)

        # İlerleme göstergesi
        em_so_far = (correct / total) * 100
        status = "✅" if em == 1 else "❌"
        print(f"  [{i+1:3d}/{eval_samples}] {status} EM={em_so_far:.1f}% | {question[:60]}...")

        # API rate limit: istekler arası bekleme
        if i < eval_samples - 1:
            time.sleep(delay_between_requests)

    em_score = (correct / total) * 100 if total > 0 else 0

    print(f"\n--- SONUÇ: {baseline_type.upper()} ({schema_format}) ---")
    print(f"  Exact Match: {correct}/{total} = {em_score:.1f}%")

    return {
        "baseline_type": baseline_type,
        "schema_format": schema_format,
        "model_name": model_name,
        "num_samples": total,
        "correct": correct,
        "em_score": em_score,
        "predictions": predictions,
        "error_examples": errors[:10],  # İlk 10 hata örneği
        "timestamp": datetime.now().isoformat(),
    }


def run_full_evaluation(args):
    """
    Tüm baseline deneyleri çalıştırır ve sonuçları kaydeder.
    """
    print("=" * 60)
    print("SPIDER VALIDATION SET — BASELINE EVALUATION")
    print("=" * 60)

    # Verisetini yükle
    print("\nSpider veriseti yükleniyor...")
    dataset = load_dataset("xlangai/spider")
    print(f"  Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])}")

    # OpenRouter istemcisini hazırla
    print(f"\nOpenRouter model ayarlandı: {args.model}...")
    client = get_openrouter_client()

    # Sonuçlar
    all_results = {}
    summary = {}

    # ─── Baseline 1: Zero-shot ───────────────────────────────────────────
    print("\n\n" + "#" * 60)
    print("# BASELINE 1: ZERO-SHOT PROMPTING")
    print("#" * 60)

    zs_results = evaluate_baseline(
        dataset, client, args.model, "zero_shot", args.schema_format,
        args.num_samples, args.delay
    )
    all_results["zero_shot"] = zs_results
    summary["Zero-shot Prompting"] = {
        "correct": zs_results["correct"],
        "total": zs_results["num_samples"],
    }

    # ─── Baseline 2: Few-shot ────────────────────────────────────────────
    print("\n\n" + "#" * 60)
    print("# BASELINE 2: FEW-SHOT PROMPTING (3-shot)")
    print("#" * 60)

    fs_results = evaluate_baseline(
        dataset, client, args.model, "few_shot", args.schema_format,
        args.num_samples, args.delay
    )
    all_results["few_shot"] = fs_results
    summary["Few-shot Prompting (3-shot)"] = {
        "correct": fs_results["correct"],
        "total": fs_results["num_samples"],
    }

    # ─── Sonuç tablosu ──────────────────────────────────────────────────
    print_results_table(summary)

    # ─── Sonuçları kaydet ────────────────────────────────────────────────
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Detaylı sonuçlar
    results_path = os.path.join(results_dir, "baseline_results.json")
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
                "correct": zs_results["correct"],
                "total": zs_results["num_samples"],
            },
            "few_shot": {
                "em_score": fs_results["em_score"],
                "correct": fs_results["correct"],
                "total": fs_results["num_samples"],
            },
        },
        "zero_shot_predictions": zs_results["predictions"],
        "few_shot_predictions": fs_results["predictions"],
        "zero_shot_errors": zs_results["error_examples"],
        "few_shot_errors": fs_results["error_examples"],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nDetaylı sonuçlar kaydedildi: {results_path}")

    # Özet sonuçlar (rapor için)
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(save_data["results"], f, indent=2)
    print(f"Özet sonuçlar kaydedildi: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Baseline Evaluation on Spider"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Değerlendirilecek örnek sayısı (default: 50)"
    )
    parser.add_argument(
        "--schema_format", type=str, default="format_a",
        choices=["format_a", "format_b", "format_c"],
        help="Şema serileştirme formatı (default: format_a)"
    )
    parser.add_argument(
        "--model", type=str, default="baidu/cobuddy:free",
        help="OpenRouter model adı (default: baidu/cobuddy:free)"
    )
    parser.add_argument(
        "--delay", type=float, default=4.0,
        help="API istekleri arası bekleme süresi - saniye (default: 1.5)"
    )

    args = parser.parse_args()
    run_full_evaluation(args)


if __name__ == "__main__":
    main()
