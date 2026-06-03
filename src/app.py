import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MERGED_MODEL_DIR = "models/sql-qwen-coder-1.5b-merged"

print(" - Model is loading (4-bit Quantization)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_DIR,
    quantization_config=bnb_config,
    device_map={"": 0}, 
)
print(" - Model is ready!")

def generate_sql(schema, question):
    if not schema.strip() or not question.strip():
        return "Please provide both the database schema and the question."

    prompt = (
        f"### Instruction:\n"
        f"Convert the natural language question into a SQL query based on the provided database schema.\n"
        f"Strict Rules:\n"
        f"1. Use ONLY the table and column names explicitly defined in the schema. Do not pluralize table names.\n"
        f"2. Do not hallucinate columns unless they exist in the schema.\n"
        f"3. Use JOIN only when columns from multiple tables are required.\n\n"
        f"### Input:\n"
        f"Schema:\n{schema}\n"
        f"Question: {question}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    # Clean the generated SQL query
    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    sql_result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    sql_result = sql_result.split(";")[0].replace('"', "'")

    return sql_result

# --- GRADIO INTERFACE ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Local Text-to-SQL Assistant (Qwen-1.5B)")
    gr.Markdown("This tool converts natural language questions into SQL queries based on the provided database schema. All operations are performed on the local GPU (3050 Ti).")

    with gr.Row():
        with gr.Column(scale=1):
            schema_input = gr.Textbox(
                lines=10, 
                label="Database Schema",
                placeholder="e.g., CREATE TABLE student (stuid INT, fname VARCHAR, age INT);"
            )
            question_input = gr.Textbox(
                lines=2, 
                label="Your Question (English)",
                placeholder="e.g., Find the first name of students who are older than 20."
            )
            submit_btn = gr.Button("Generate SQL 🚀", variant="primary")
            
        with gr.Column(scale=1):
            sql_output = gr.Code(
                language="sql", 
                label="Generated SQL Query", 
                interactive=False,
            )

    submit_btn.click(
        fn=generate_sql, 
        inputs=[schema_input, question_input], 
        outputs=sql_output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())