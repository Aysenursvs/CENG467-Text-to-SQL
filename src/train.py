"""
train.py — Text-to-SQL modeli için Instruction Tuning (LoRA/PEFT) eğitim scripti.

Bu script:
  1. Hazırlanan JSONL veri setini yükler.
  2. Açık kaynaklı bir LLM'i 4-bit (bellek tasarrufu için) yükler.
  3. LoRA (Low-Rank Adaptation) adaptörlerini yapılandırır.
  4. SFTTrainer (Supervised Fine-Tuning) ile modeli eğitir.
  5. Eğitilen adaptör ağırlıklarını 'models/' klasörüne kaydeder.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ─── 1. AYARLAR VE YAPILANDIRMA ──────────────────────────────────────────────
# Modeli değiştirmek isterseniz burayı güncelleyebilirsiniz (Örn: meta-llama/Meta-Llama-3-8B)
MODEL_NAME = "mistralai/Mistral-7B-v0.1" 
DATASET_PATH = "data/train_formatted.jsonl"
OUTPUT_DIR = "models/sql-mistral-lora"

def formatting_prompts_func(example):
    """
    Alpaca formatındaki veriyi modelin eğitim sırasında okuyacağı tekil bir metne çevirir.
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n### Input:\n{example['input'][i]}\n\n### Response:\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

def main():
    print("=" * 60)
    print("🚀 Text-to-SQL Model Eğitimi (Instruction Tuning) Başlıyor!")
    print("=" * 60)

    # ─── 2. VERİ SETİNİ YÜKLEME ───────────────────────────────────────────────
    print(f"\n[1/5] Eğitim verisi yükleniyor: {DATASET_PATH}")
    dataset = load_dataset("json", data_files={"train": DATASET_PATH})

    # ─── 3. MODEL VE TOKENIZER YÜKLEME (4-BIT QUANTIZATION) ───────────────────
    print(f"\n[2/5] Tokenizer ve Model ({MODEL_NAME}) 4-bit olarak yükleniyor...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Padding için eos_token kullanıyoruz

    # Modeli ekran kartına (GPU) sığdırmak için 4-bit sıkıştırma (Quantization) ayarı
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Modeli otomatik olarak müsait GPU'ya yay
        trust_remote_code=True
    )
    
    # Modeli eğitim için hazırlar
    model = prepare_model_for_kbit_training(model)

    # ─── 4. LORA (PEFT) YAPILANDIRMASI ────────────────────────────────────────
    print("\n[3/5] LoRA (Low-Rank Adaptation) adaptörleri kuruluyor...")
    # Sadece belirli katmanları eğiterek devasa bellek (RAM/VRAM) tasarrufu sağlıyoruz
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] 
    )

    # ─── 5. EĞİTİM ARGÜMANLARI (TRAINING ARGS) ────────────────────────────────
    print("\n[4/5] Eğitim parametreleri ayarlanıyor...")
    training_args = SFTConfig(               # <-- TrainingArguments yerine SFTConfig oldu
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,       
        gradient_accumulation_steps=4,       
        learning_rate=2e-4,                  
        logging_steps=10,                    
        max_steps=200,                       
        optim="paged_adamw_8bit",
        fp16=True,                           
        save_strategy="steps",
        save_steps=50,
        max_seq_length=1024,                 # <-- max_seq_length buraya taşındı!
    )

    # ─── 6. SFT TRAINER İLE EĞİTİMİ BAŞLATMA ──────────────────────────────────
    print("\n[5/5] SFTTrainer başlatılıyor...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
        # max_seq_length buradan silindi
    )

    print("\n🔥 Eğitim başlatılıyor! (Bu işlem donanıma göre saatler sürebilir)...\n")
    trainer.train()

    # ─── 7. MODELİ KAYDETME ───────────────────────────────────────────────────
    print(f"\n✅ Eğitim tamamlandı! Adaptörler '{OUTPUT_DIR}' klasörüne kaydediliyor...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("🚀 Her şey hazır. Model test edilmeyi bekliyor!")

if __name__ == "__main__":
    main()