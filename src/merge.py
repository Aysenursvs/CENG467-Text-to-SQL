import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
LORA_DIR   = "models/sql-qwen-coder-1.5b-lora"
MERGED_DIR = "models/sql-qwen-coder-1.5b-merged"

def main():
    print(" - Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print(" - Loading base model onto CPU...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu", 
        trust_remote_code=True
    )
    
    print(" - Integrating LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    
    print(" - Merging weights (Merge & Unload)... This may take 1-2 minutes.")
    merged_model = model.merge_and_unload()
    
    print(f" - Saving merged model to '{MERGED_DIR}'...")
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(" - Merge completed successfully!")

if __name__ == "__main__":
    main()