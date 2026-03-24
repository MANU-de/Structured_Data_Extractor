## Structured Data Extractor – Invoice Agent Service

This repository contains an end‑to‑end pipeline for training, deploying and consuming a **LLM‑based invoice extractor**.  
The model is deployed as a **Modal** web service and can be exercised via a simple **Python client**.

---
## 🛠 Environment & Setup

### 1. Prerequisites
- **Python**: 3.10 or higher is required.
- **Hardware**: Local installation only requires a CPU for the client. Deployment/Training requires an NVIDIA GPU (T4 or better).

### 2. Installation & Virtual Environment
It is highly recommended to use a virtual environment to keep your project dependencies isolated.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install all project dependencies (Core, Client, and Deployment)
pip install -r requirements.txt
```
---
## Authentication & Infrastructure Setup

This project uses Modal for serverless GPU deployment. Once the library is installed, you must link your local environment to your Modal account.

```bash
# Authenticate with Modal
modal auth signup
```
---

## Required Secrets & API Keys

The Modal deployment uses **three secrets** (see `deploy/deploy.py`):

- **Hugging Face token**: stored in Modal as `my-huggingface-secret`
- **Weights & Biases API key**: stored in Modal as `my-wandb-secret`
- **Deployment API auth key**: stored in Modal as `my-deployment-secrets`

### 1. Create the Modal Secrets

Make sure you have:

- A Hugging Face access token with permission to read the model  
  `manuelaschrittwieser/llama-3-invoice-extractor-merged`
- A Weights & Biases API key with access to the project:
  - Project name: `invoice-extractor-production`
- An **API auth key** of your choice (any non‑empty string) that will be required by the public endpoint

Then, from your local machine (after installing `modal` and logging in with `modal token new`):

```bash
modal secret create my-huggingface-secret HF_TOKEN=xxxxxx
modal secret create my-wandb-secret WANDB_API_KEY=yyyyyy
modal secret create my-deployment-secrets MY_API_AUTH_KEY=zzzzzz
```

Replace `xxxxxx`/`yyyyyy`/`zzzzzz` with your actual keys.

> **Security model**:  
> The web endpoint validates an `Authorization: Bearer <API_KEY>` header against `MY_API_AUTH_KEY` from the `my-deployment-secrets` Modal secret.  
> Requests without a valid key receive `401 Unauthorized`.

---

## Deploying the Invoice Agent Service (Modal)

The deployment entrypoint is `deploy/deploy.py`.  
It defines:

- A GPU‑backed `Model` class that loads the fine‑tuned model from Hugging Face.
- A `@modal.web_endpoint` called `api` that exposes a POST endpoint accepting:
  - JSON body: `{"text": "<invoice free‑text>"}`  
  - Response JSON:
    - `data`: extracted invoice fields as JSON
    - `model`: model identifier
    - `status`: status string

### 1. Set up Modal

```bash
pip install modal
modal token new
```

Follow the prompts in your browser to authenticate.

### 2. Deploy the Service

From the project root:

```bash
cd deploy
modal deploy deploy.py
```

Modal will:

- Build the image (with all ML dependencies).
- Allocate GPU resources as configured in `deploy.py`.
- Output the public URL of the web endpoint.

> The URL typically looks like:
> `https://<username>--your-app-name.modal.run`

### 3. Verify Deployment

After deployment, you can send a quick manual test using `curl` (replace the URL with your own):

```bash
curl -X POST \
  URL = "https://your-app-name.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"text": "I bought 2 monitors at Dell for 800 dollars yesterday. Paid in USD."}'
```

You should receive a JSON response containing the extracted fields.

---

## Running the Client & Testing the Deployment

The client test script is `client/test_api.py`. It:

- Sends HTTP POST requests to the deployed Modal endpoint.
- Contains a small **evaluation suite** to validate extraction quality.

### 1. Configure the Endpoint URL

By default, the script expects an endpoint URL pointing to your deployed Modal service.  
You should set this to **your own endpoint**, not a shared or public URL.

```python
URL = "https://your-app-name.modal.run"
```

If your Modal deployment exposes a different URL, update the `URL` constant at the top of `client/test_api.py`.

### 1.1 Configure the Client API Key (`.env`)

The client uses a **Bearer token** to authenticate against the Modal endpoint:

- Create a local `.env` file in the project root:

```bash
echo 'MY_API_AUTH_KEY=zzzzzz' > .env
```

- The value for `MY_API_AUTH_KEY` **must match** the value used when creating the Modal secret `my-deployment-secrets`.
- The client will automatically:
  - Load this value via `python-dotenv`
  - Send it as `Authorization: Bearer <MY_API_AUTH_KEY>` on every request

### 2. Run the Automated Evaluation Suite

From the project root:

```bash
python client/test_api.py
```

This will:

- Call the API for each test case in `TEST_SUITE`.
- Print the pretty‑formatted JSON response.
- Run a basic validation (e.g., vendor field match) and log the outcome.

### 3. Run a Single Manual Test

You can also invoke the client with an arbitrary invoice description:

```bash
python client/test_api.py "Spent 45.50 EUR at Mario's Pizza for 3 pizzas on Oct 12th."
```

This sends the given text to the API and prints the model’s structured JSON output.

---

## Local Tools (Optional)

The repo also includes some helper scripts under:

- `local_tools/cli_extract.py`
- `local_tools/run_inference.py`

These can be used to run the model or utilities locally (e.g., for experimentation or debugging) once you have the model and dependencies set up. Their usage is optional for deployment and for the standard client evaluation flow.

---

## Summary

- **Environment setup**: install Python 3.10+, dependencies from `requirements.txt`, `modal`, and `httpx`.
- **Secrets**: configure Hugging Face and Weights & Biases keys as Modal secrets.
- **Deployment**: `modal deploy deploy.py` under `deploy/` to get a public API endpoint.
- **Client tests**: use `python client/test_api.py` to run the full evaluation suite or pass a text argument for manual testing.
