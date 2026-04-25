import requests
import json
import csv
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Please check your .env file.")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-4-26b-a4b-it:free"
ALLOWED_SCOPE = ["Apple", "Banana", "Orange", "Strawberry"]

def clean_value_with_llm(value_to_clean, scope):
    scope_string = ", ".join(scope)

    system_instruction = f"""You are an automated data ingestion assistant. 
Your allowed scope of values is: [{scope_string}].

Evaluate the user's input and respond strictly in JSON format with exactly three keys: "state", "message", and "value".
Do not wrap your response in markdown blocks (e.g., ```json).

Rules:
1. "state": "acceptance"
   - Use when: The input matches something in the scope (allowing for typos, case differences, or clear synonyms).
   - "message": MUST say "This value is good and should be named [Standardized Name]".
   - "value": MUST be the exact Standardized Name from the scope.

2. "state": "decline"
   - Use when: The input is garbage, a completely different category, or invalid data.
   - "message": MUST explain why it shouldn't be ingested.
   - "value": MUST be an empty string "".

3. "state": "suggest"
   - Use when: The input is a valid item of the same category (e.g., a fruit) but is NOT in the scope.
   - "message": MUST say "This value should be added to our scope".
   - "value": MUST be the cleaned, properly capitalized name of the suggested new item.
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Value to evaluate: {value_to_clean}"}
        ],
        "temperature": 0.1
    }

    max_retries = 3
    backoff_time = 5

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)

            if response.status_code == 429:
                print(
                    f"⚠️ Rate limited (429). Waiting {backoff_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(backoff_time)
                backoff_time *= 2
                continue

            response.raise_for_status()

            raw_output = response.json()["choices"][0]["message"]["content"].strip()

            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()

            return json.loads(raw_output)

        except requests.exceptions.RequestException as e:
            return {"state": "error", "message": f"API Request failed: {e}", "value": ""}
        except json.JSONDecodeError:
            return {"state": "error", "message": f"Failed to parse JSON. Raw output: {raw_output}", "value": ""}

    return {"state": "error", "message": "Failed due to repeated 429 Rate Limit errors.", "value": ""}

def process_csv(filepath, scope):
    print(f"Processing data from {filepath}...\n")

    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            raw_value = row.get("raw_value")
            if not raw_value:
                continue

            print(f"Evaluating: '{raw_value}'")
            result = clean_value_with_llm(raw_value, scope)

            print(json.dumps(result, indent=2))
            print("-" * 40)

            time.sleep(2)

if __name__ == "__main__":
    csv_filename = "input_data.csv"
    process_csv(csv_filename, ALLOWED_SCOPE)