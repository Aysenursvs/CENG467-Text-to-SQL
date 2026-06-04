"""
train_mistral.py - Instruction tuning (LoRA/PEFT) training script for Text-to-SQL.

This script:
    1. Loads the prepared JSONL dataset.
    2. Loads an open-source LLM in 4-bit mode to save memory.
    3. Configures LoRA (Low-Rank Adaptation) adapters.
    4. Trains the model with SFTTrainer (supervised fine-tuning).
    5. Saves the trained adapter weights to the output folder.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# --- 1. SETTINGS AND CONFIGURATION ---
# Update this if you want to train a different base model.
MODEL_NAME = "mistralai/Mistral-7B-v0.1" 
DATASET_PATH = "data/train_formatted.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/sql-mistral-lora"

def formatting_prompts_func(example):
    """
    Convert Alpaca-style fields into a single training text sequence.
    """
    def format_one(instruction, input_text, output_text):
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}</s>"
        )

    if isinstance(example["instruction"], str):
        return format_one(example["instruction"], example["input"], example["output"])

    output_texts = []
    for i in range(len(example["instruction"])):
        output_texts.append(
            format_one(example["instruction"][i], example["input"][i], example["output"][i])
        )
    return output_texts

def main():
    print("=" * 60)
    print("Text-to-SQL Model Training (Instruction Tuning) Starting")
    print("=" * 60)

    # --- 2. DATASET LOADING ---
    print(f"\n[1/5] Loading training data: {DATASET_PATH}")
    dataset = load_dataset("json", data_files={"train": DATASET_PATH})
    # Keep a small eval split for monitoring training quality.
    split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"  Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    # --- 3. MODEL AND TOKENIZER LOADING (4-BIT QUANTIZATION) ---
    print(f"\n[2/5] Loading tokenizer and model ({MODEL_NAME}) in 4-bit...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Use EOS token for padding to avoid adding a new token.
    tokenizer.pad_token = tokenizer.eos_token

    # Quantization settings to fit the model into GPU memory.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically place layers on available GPUs.
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training.
    model = prepare_model_for_kbit_training(model)

    # --- 4. LORA (PEFT) CONFIGURATION ---
    print("\n[3/5] Setting up LoRA (Low-Rank Adaptation) adapters...")
    # Train only selected projection layers to reduce memory usage.
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
        ] 
    )

    # --- 5. TRAINING ARGUMENTS ---
    print("\n[4/5] Configuring training parameters...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,       # Reduce if GPU memory is limited.
        gradient_accumulation_steps=4,       # Virtual batch size (4 x 4 = 16).
        learning_rate=2e-4,                  # Learning rate.
        logging_steps=10,                    # Log every N steps.
        num_train_epochs=2,                  # Full epoch-based training.
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=False,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=25,
        max_length=1024,
    )

    # --- 6. START TRAINING WITH SFTTRAINER ---
    print("\n[5/5] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    print("\nTraining started (may take hours depending on hardware)...\n")
    # Set resume_from_checkpoint to a valid path if you want to resume.
    trainer.train(resume_from_checkpoint="/content/drive/MyDrive/sql-mistral-lora/checkpoint-75")

    # --- 7. SAVE ADAPTERS ---
    print(f"\nTraining complete. Saving adapters to '{OUTPUT_DIR}'...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("All set. The model is ready for evaluation.")

if __name__ == "__main__":
    main()
