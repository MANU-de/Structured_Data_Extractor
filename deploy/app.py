import gradio as gr
import torch
from unsloth import FastLanguageModel
import json
import re

# --- CONFIGURATION ---
MODEL_NAME = "manuelaschrittwieser/llama-3-invoice-extractor" 

# --- MODEL LOADING ---
# This runs when the Hugging Face Space starts
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# --- HELPER FUNCTIONS ---
def clean_json_string(raw_output):
    """Cleans potential markdown code blocks from the LLM output."""
    cleaned = re.sub(r"```json\n?|```", "", raw_output).strip()
    return cleaned

def process_invoice_logic(text):
    """The Agentic workflow: Extract -> Parse -> Calculate -> Log"""
    logs = "🚀 Agent started...\n"
    
    # 1. Prompt Construction
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Extract invoice details into JSON.

### Input:
{text}

### Response:
"""
    
    # 2. Inference
    logs += "🧠 Model is analyzing text...\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 3. Extract JSON part from response
    try:
        raw_json_str = response.split("### Response:\n")[-1]
        cleaned_json = clean_json_string(raw_json_str)
        data = json.loads(cleaned_json)
        logs += "✅ Data extraction successful.\n"
    except Exception as e:
        return {"error": "Failed to parse JSON"}, f"❌ Error: {str(e)}"

    # 4. Agentic Logic (Calculations)
    logs += "📊 Performing post-processing calculations...\n"
    
    total = data.get("total", 0)
    currency = data.get("currency", "USD")
    
    # Logic: Calculate 15% service tax and grand total
    if isinstance(total, (int, float)) and total > 0:
        tax = total * 0.15
        grand_total = total + tax
        logs += f"📝 Applied 15% Tax: {tax:.2f} {currency}\n"
        logs += f"💰 Calculated Grand Total: {grand_total:.2f} {currency}\n"
        data["agent_calculated_tax"] = round(tax, 2)
        data["agent_grand_total"] = round(grand_total, 2)
    else:
        logs += "⚠️ No numeric total found; skipping calculations.\n"

    logs += "🏁 Task complete."
    return data, logs

# --- GRADIO UI DESIGN ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Invoice Agent") as demo:
    gr.Markdown("# 📑 AI Invoice Agent")
    gr.Markdown("This agent uses a **Fine-Tuned Llama-3 model** to extract structured data from messy text and perform business logic.")
    
    with gr.Row():
        # Left Side: Input
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                label="Messy Invoice Text",
                placeholder="e.g. Spent 200 dollars on a new monitor at Amazon on Jan 1st...",
                lines=8
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Analyze & Process", variant="primary")
            
            gr.Examples(
                examples=[
                    ["Ordered 3x MacBooks from Apple Inc for $6000 on Oct 12th. Paid via Credit Card."],
                    ["Lunch at Mario's Pizza, total 45.50 Euro. Included 2 pizzas and drinks."],
                    ["Best Buy: 1 Sony Headphones, $299. Nov 24, 2023."]
                ],
                inputs=input_box
            )

        # Right Side: Output
        with gr.Column(scale=1):
            output_json = gr.JSON(label="Extracted JSON Data")
            status_logs = gr.Textbox(
                label="Agent Execution Logs", 
                lines=6, 
                interactive=False
            )

    # --- BUTTON ACTIONS ---
    submit_btn.click(
        fn=process_invoice_logic,
        inputs=input_box,
        outputs=[output_json, status_logs]
    )
    
    clear_btn.click(lambda: [None, None, ""], outputs=[input_box, output_json, status_logs])

# --- LAUNCH ---
if __name__ == "__main__":
    demo.launch()