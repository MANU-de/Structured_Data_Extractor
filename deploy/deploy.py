import modal
import json
import os
from pydantic import BaseModel
from fastapi import Header, HTTPException 

# 1. Define the Container Image 
# This is essentially a Dockerfile written in Python.
image = (
    modal.Image.debian_slim()
    .pip_install(
        "unsloth", 
        "torch", 
        "transformers", 
        "accelerate", 
        "bitsandbytes", 
        "fastapi", 
        "pydantic",
        "wandb"
    )
)

app = modal.App("invoice-agent-service")

# 2. Define Request/Response Schemas (API Guardrails)
class InvoiceRequest(BaseModel):
    text: str

# 3. The Model Class
@app.cls(
    gpu="T4", 
    image=image,
    # --- AUTO-SCALING CONFIGURATION ---
    min_containers=0,              # Scale to zero when not in use (Saves money)
    max_containers=10,             # Can scale up to 10 GPUs simultaneously
    scaledown_window=60,   # Keep the GPU warm for 60 seconds after a request
    # ----------------------------------
    # SECURE METHOD:
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("my-wandb-secret"),
        modal.Secret.from_name("my-deployment-secrets")
    ]
)
class Model:
    @modal.enter()
    def load_model(self):
        """This runs once when the container starts up"""
        import wandb
        from unsloth import FastLanguageModel
        
        
        
        # Initialize W&B for Production Monitoring
        wandb.init(project="invoice-extractor-production", job_type="inference")
        
        # Load the merged model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="manuelaschrittwieser/llama-3-invoice-extractor-merged",
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    @modal.method()
    def generate(self, text: str):
        import wandb
        import re
        
        # 1. Construct the exact prompt used in training
        prompt = f"### Instruction:\nExtract invoice details into JSON.\n\n### Input:\n{text}\n\n### Response:\n"
        
        # 2. Tokenize and Generate
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=128, use_cache=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 3. Clean and parse the JSON
        raw_json_str = response.split("### Response:\n")[-1]
        # Remove any potential markdown code blocks (e.g., ```json ... ```)
        cleaned_json = re.sub(r"```json\n?|```", "", raw_json_str).strip()
        
        # 4. Log to Weights & Biases 
        wandb.log({
            "input_text": text,
            "output_json": cleaned_json,
            "status": "success"
        })
        
        return json.loads(cleaned_json)

# 4. The Web Endpoint 
@app.function(
    image=image, 
    
    secrets=[modal.Secret.from_name("my-deployment-secrets")] 
)
    
@modal.web_endpoint(method="POST")
def api(request: InvoiceRequest, authorization: str = Header(None)):
    valid_key = os.environ.get("MY_API_AUTH_KEY")
    
    # TEMPORARY LOGGING (Remove after testing)
    print(f"DEBUG: Received Header: {authorization}")
    print(f"DEBUG: Expected Header: Bearer {valid_key}")
 
    
    # 2. Check if the key was provided and if it matches
    if authorization != f"Bearer {valid_key}":
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Invalid or missing API Key"
        )

    # 3. If valid, proceed to call the GPU model
    model_instance = Model()
    extracted_data = model_instance.generate.remote(request.text)
    
    # engineering practice: add metadata to the response
    return {
        "data": extracted_data,
        "model": "llama-3-8b-invoice-v2",
        "status": "completed"
    }