"""
evaluate.py — Eğitilmiş Text-to-SQL modelini lokalde test etme scripti.
"""

import os
import sqlite3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- AYARLAR ---
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_MODEL_DIR = "models/sql-mistral-lora"
DB_DIR = "data/database"

def execute_sql(db_id, sql_query):
    """Verilen SQL sorgusunu veritabanında çalıştırır."""
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

def main():
    print("=" * 70)
    print("🚀 Text-to-SQL Modeli Lokalde Ayağa Kaldırılıyor...")
    print("=" * 70)

    # 4-Bit Sıkıştırma (Windows'ta ekran kartı belleğini aşmamak için kritik)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("[1/3] Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("[2/3] Temel model (Mistral-7B) 4-bit olarak yükleniyor...")
    # device_map="auto" sayesinde VRAM yetmezse sistem RAM'inden destek alır
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto" 
    )

    print("[3/3] Eğittiğin LoRA adaptörleri modele entegre ediliyor...")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    
    print("\n✅ Sistem Hazır! Soru soruluyor...\n")

    # --- TEST SENARYOSU ---
    test_db = "department_management"
    schema = "CREATE TABLE department (Department_ID number, Name text, Creation text, Ranking number, Budget_in_Billions number, Num_Employees number);"
    question = "How many departments are there?"
    hedef_sql = "SELECT count(*) FROM department"
    
    prompt = (
        "### Instruction:\n"
        "You are an expert SQL developer. Your task is to translate the given natural language question into a valid executable SQL query.\n\n"
        "### Input:\n"
        f"Database: {test_db}\nSchema:\n{schema}\n\nQuestion: {question}\n\n"
        "### Response:\n"
    )
    
    # Soruyu ekran kartına (cuda) gönderiyoruz
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(f"Soru: {question}")
    print("🧠 Model düşünüyor...")
    
    outputs = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
    uretilen_metin = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Modelin cevabından sadece SQL'i çekiyoruz
    uretilen_sql = uretilen_metin.split("### Response:")[-1].split(";")[0].strip()
    
    print("-" * 60)
    print(f"🎯 Beklenen Hedef SQL : {hedef_sql}")
    print(f"🤖 Modelin Ürettiği SQL: {uretilen_sql}")
    print("-" * 60)
    
    print("\n[Execution] Sorgular veritabanında çalıştırılıyor...")
    hedef_sonuc = execute_sql(test_db, hedef_sql)
    model_sonuc = execute_sql(test_db, uretilen_sql)
    
    print(f"Hedef Çıktı : {hedef_sonuc}")
    print(f"Model Çıktı : {model_sonuc}")
    
    if hedef_sonuc == model_sonuc and "HATA" not in str(model_sonuc):
        print("\n🏆 SONUÇ: BAŞARILI! (Execution Accuracy: 1)")
    else:
        print("\n❌ SONUÇ: BAŞARISIZ! Veriler uyuşmuyor.")

if __name__ == "__main__":
    main()