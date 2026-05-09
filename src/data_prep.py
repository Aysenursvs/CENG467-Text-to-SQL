import os

from datasets import load_dataset

# 1. Spider verisetini Hugging Face üzerinden indiriyoruz
print("Spider veriseti indiriliyor/yükleniyor...")
dataset = load_dataset("spider")

# 2. Verisetinin bölümlerini (train/validation) ve boyutlarını yazdırıyoruz
print("\n--- VERİSETİ BOYUTLARI ---")
print(f"Eğitim (Train) seti örnek sayısı: {len(dataset['train'])}")
print(f"Doğrulama (Validation) seti örnek sayısı: {len(dataset['validation'])}")


# 3. Verisetinden ilk örneği çekip yapısını inceliyoruz
sample = dataset["train"][0]

print("\n--- İLK ÖRNEK İNCELEMESİ ---")
print("Veritabanı Adı (db_id):", sample["db_id"])
print("Doğal Dil Sorusu (question):", sample["question"])
print("Hedef SQL Sorgusu (query):", sample["query"])

# 4. Girdi formatının tam özelliklerini (features) görmek için
print("\n--- VERİ FORMATI (FEATURES) ---")
for key in sample.keys():
    print(f"- {key}: {type(sample[key])}")

# 5. Zero-Shot Prompt Taslağı Oluşturma (Baseline 1 için)
def create_zero_shot_prompt(question, db_id, schema_info=""):
    """
    Doğal dil sorusunu ve veritabanı şemasını alıp model için prompt oluşturur.
    """
    prompt = f"""You are an expert SQL developer. Your task is to translate the given natural language question into a valid executable SQL query.

Database Name: {db_id}
Database Schema: 
{schema_info}

Question: {question}
SQL Query:"""
    return prompt

# İlk örneğimiz için örnek bir şema bilgisi (Şimdilik manuel yazıyoruz, ileride otomatize edeceğiz)
# Gerçek projede Spider verisetindeki 'tables.json' dosyasından bu bilgiyi çekeceğiz.
sample_schema = "- table: department, columns: [Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees]\n- table: head, columns: [head_ID, name, born_state, age]"

# Promptu oluşturup ekrana basalım
baseline_prompt = create_zero_shot_prompt(sample["question"], sample["db_id"], sample_schema)

print("\n--- BASELINE 1: ZERO-SHOT PROMPT TASLAĞI ---")
print(baseline_prompt)

from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[{"role": "user", "content": "Merhaba"}],
)

print(response.choices[0].message.content)

def get_sql_prediction_api(prompt):
    try:
        # temperature=0 vermek çok önemli! SQL gibi kesin mantık gerektiren 
        # görevlerde modelin halüsinasyon yapmasını (rastgelelik) engelleriz.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Veya elinizde varsa gpt-4o-mini
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        # Sadece SQL sorgusunu alıp sağdaki soldaki boşlukları temizliyoruz
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API Hatası: {e}"

# 7. Exact Match (Birebir Eşleşme) Skoru Hesaplama Fonksiyonu
def calculate_exact_match(prediction, target):
    # Basit bir temizleme (boşlukları ve büyük/küçük harfleri eşitleme)
    pred_clean = prediction.strip().lower().replace(";", "")
    target_clean = target.strip().lower().replace(";", "")
    return 1 if pred_clean == target_clean else 0

# Test için ilk 5 örneği çalıştıralım
print("\n--- LLM API İLE İLK 5 ÖRNEK İÇİN TEST BAŞLIYOR ---")
total_em = 0
num_samples = 5

for i in range(num_samples):
    sample = dataset["train"][i]
    # Şimdilik şema kısmını boş geçiyoruz
    prompt = create_zero_shot_prompt(sample["question"], sample["db_id"], schema_info="") 
    
    prediction = get_sql_prediction_api(prompt)
    target_sql = sample["query"]
    
    em_score = calculate_exact_match(prediction, target_sql)
    total_em += em_score
    
    print(f"\nSoru {i+1}: {sample['question']}")
    print(f"Hedef SQL: {target_sql}")
    print(f"Modelin Çıktısı:\n{prediction}")
    print(f"Eşleşme: {'BAŞARILI' if em_score == 1 else 'BAŞARISIZ'}")

accuracy = (total_em / num_samples) * 100
print(f"\n--- SONUÇ ---")
print(f"Zero-Shot LLM Baseline Exact Match Skoru (İlk 5 örnek): %{accuracy}")
