import httpx
import json
import time
import os
import sys
from dotenv import load_dotenv

# Load local secrets from .env
load_dotenv()

# --- CONFIGURATION ---
# 1. Replace with your actual Modal URL
URL = "https://<username>--your-app-name.modal.run"

# 2. Get the API key from your .env file
API_KEY = os.getenv("MY_API_AUTH_KEY")

# --- EVALUATION SUITE (Rubric Requirement) ---
TEST_SUITE = [
    {
        "name": "Standard Laptop Purchase",
        "input": "I bought 2 monitors at Dell for 800 dollars yesterday. Paid in USD.",
        "expected": {"vendor": "Dell", "total": 800.0, "quantity": 2}
    },
    {
        "name": "European Pizza Receipt",
        "input": "Spent 45.50 EUR at Mario's Pizza for 3 pizzas on Oct 12th.",
        "expected": {"vendor": "Mario's Pizza", "total": 45.5, "quantity": 3}
    },
    {
        "name": "Apple Store Shorthand",
        "input": "Apple Store: 1x MacBook Pro, $2499.00. Jan 5th 2024.",
        "expected": {"vendor": "Apple Store", "total": 2499.0, "quantity": 1}
    }
]

def test_invoice_agent(text, test_name="Manual Test"):
    """
    Client-side logic to interact with the authenticated Modal API.
    """
    if not API_KEY:
        print("❌ ERROR: MY_API_AUTH_KEY not found in .env file.")
        return

    print(f"\n--- 🚀 Running: {test_name} ---")
    print(f"📥 Input: '{text}'")
    
    # Define Headers with Bearer Token (Security Implementation)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Configure timeout for Cold Starts (120s) and allow redirects (Modal 303s)
    timeout = httpx.Timeout(120.0, read=None)
    
    start_time = time.time()
    
    try:
        # use follow_redirects=True to handle Modal's internal load balancing
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(URL, json={"text": text}, headers=headers)
            
            # Error Handling for Authentication
            if response.status_code == 401:
                print("❌ UNAUTHORIZED: Your API Key is incorrect or missing.")
                return None
            
            # General Error Handling
            response.raise_for_status()
            
            result = response.json()
            latency = time.time() - start_time
            
            print(f"✅ SUCCESS ({latency:.2f}s)")
            print(json.dumps(result, indent=4))
            return result

    except httpx.ReadTimeout:
        print("❌ TIMEOUT: The GPU is still warming up. Please try again in 30 seconds.")
    except httpx.HTTPStatusError as e:
        print(f"❌ SERVER ERROR: {e.response.status_code}")
        print(f"Details: {e.response.text}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
    
    return None

def run_full_evaluation():
    """
    Executes the automated test suite for project evaluation.
    """
    print("===============================================")
    print("🔍 STARTING DEPLOYMENT EVALUATION SUITE")
    print("===============================================")
    
    for test in TEST_SUITE:
        test_invoice_agent(test["input"], test["name"])
        
    print("\n===============================================")
    print("🏁 EVALUATION COMPLETE")
    print("===============================================")

if __name__ == "__main__":
    # If you run 'python test_api.py "some text"', it tests that text.
    # If you just run 'python test_api.py', it runs the full evaluation suite.
    if len(sys.argv) > 1:
        test_invoice_agent(sys.argv[1])
    else:
        run_full_evaluation()