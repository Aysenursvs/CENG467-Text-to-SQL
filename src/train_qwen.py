import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig  # SFTConfig: trl>=0.12 ile geldi

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "qwen-coder-1.5b": {
        "name": "Qwen/Qwen2.5-Coder-1.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
}

ACTIVE_MODEL = "qwen-coder-1.5b"
MODEL_NAME   = MODEL_CONFIGS[ACTIVE_MODEL]["name"]
TARGET_MODS  = MODEL_CONFIGS[ACTIVE_MODEL]["target_modules"]
DATASET_PATH = "data/train_formatted.jsonl"
OUTPUT_DIR   = f"models/sql-{ACTIVE_MODEL}-lora"
MAX_SEQ_LEN  = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Format ──────────────────────────────────────────────────────────────
def formatting_prompts_func(examples):
    """Formats the input examples into a list of formatted prompts."""
    if isinstance(examples.get("instruction"), list):
        return [
            f"### Instruction:\n{examples['instruction'][i]}\n\n"
            f"### Input:\n{examples['input'][i]}\n\n"
            f"### Response:\n{examples['output'][i]}"
            for i in range(len(examples["instruction"]))
        ]
    return (
        f"### Instruction:\n{examples['instruction']}\n\n"
        f"### Input:\n{examples['input']}\n\n"
        f"### Response:\n{examples['output']}"
    )

# ─── Main Function ───────────────────────────────────────────────────────────
def main():
    print(f" - Local training started with {ACTIVE_MODEL} model!\n")

    # CUDA control
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not found! Please check your Driver/CUDA Toolkit installation.")
    print(f"✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   PyTorch {torch.__version__} | CUDA {torch.version.cuda}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files={"train": DATASET_PATH})
    print(f" -  {len(dataset['train'])} education examples loaded.")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 4-Bit Quantization ───────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print(" - Model is loading...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=TARGET_MODS,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=False,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        packing=False,
        max_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        args=sft_config,
        formatting_func=formatting_prompts_func,
    )

    print("\n - Training started...")
    trainer.train()

    print(f"\n✅ Completed! Saving to '{OUTPUT_DIR}'...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("🎉 Finished!")

if __name__ == "__main__":
    main()