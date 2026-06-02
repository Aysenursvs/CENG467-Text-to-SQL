"""
evaluate.py — Eğitilmiş Text-to-SQL modelini test etme ve Execution Accuracy hesaplama scripti.

Bu script:
  1. Eğitilmiş LoRA modelini (Mistral) yükler.
  2. Validation (Test) setindeki soruları modele sorar.
  3. Modelin ürettiği SQL'i fiziksel .sqlite veritabanında ÇALIŞTIRIR (Execution).
  4. Hedef SQL'i çalıştırır ve iki tablonun sonucunu kıyaslar.
"""

import os
import sqlite3
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- AYARLAR ---
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL_DIR = "models/sql-mistral-lora"
DB_DIR = "data/database"

def execute_sql(db_id, sql_query):
    """
    Verilen SQL sorgusunu ilgili SQLite veritabanında çalıştırır ve sonuçları döndürür.
    """
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    
    if not os.path.exists(db_path):
        return "DB_NOT_FOUND"

    try:
        # Veritabanına bağlan ve sorguyu çalıştır
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        
        # Sonuçları 'set' (küme) yapıyoruz çünkü eğer soruda "ORDER BY" (Sırala) 
        # istenmemişse, satırların hangi sırayla geldiğinin bir önemi yoktur.
        return set(result) 
    except Exception as e:
        return f"SQL_ERROR: {e}"

def main():
    print("=" * 60)
    print("🧪 Execution Accuracy Değerlendirmesi Başlıyor...")
    print("=" * 60)

    # (Buraya ilerleyen aşamada modeli yükleme ve predict (tahmin) kodlarımızı ekleyeceğiz)
    
    # --- KÜÇÜK BİR SİMÜLASYON TESTİ (Model çalışmadan önce mantığı test edelim) ---
    print("\n[Test] Execution Mantığı Kontrol Ediliyor...")
    
    test_db = "department_management"
    hedef_sql = "SELECT name FROM department"
    # Modelin ürettiği SQL (farklı yazılmış ama sonucu aynı olmalı)
    modelin_sqli = "SELECT T1.name FROM department AS T1" 
    
    hedef_sonuc = execute_sql(test_db, hedef_sql)
    model_sonuc = execute_sql(test_db, modelin_sqli)

    print(f"\nVeritabanı: {test_db}")
    print(f"Hedef SQL Sonucu: {hedef_sonuc}")
    print(f"Model SQL Sonucu: {model_sonuc}")

    if hedef_sonuc == model_sonuc:
        print("\n✅ BAŞARILI! Exact Match (Birebir Eşleşme) buna 'Yanlış' derdi ama Execution Accuracy 'Doğru' dedi!")
    else:
        print("\n❌ BAŞARISIZ! Tablolar eşleşmedi.")

if __name__ == "__main__":
    main()