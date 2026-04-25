import requests
import json
import csv
import os
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found! Please check your .env file.")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-4-26b-a4b-it:free"

def clean_value_with_llm(value_to_clean, scope):
    scope_string = ", ".join(scope)

    system_instruction = f"""You are an automated data ingestion assistant. 
Your allowed scope of values is: [{scope_string}].

Evaluate the user's input and respond strictly in JSON format with exactly three keys: "state", "message", and "value".
Do not wrap your response in markdown blocks (e.g., ```json).

Rules:
1. "state": "acceptance"
   - Use when: The input matches something in the scope (allowing for typos, case differences, or clear synonyms).
   - "message": MUST say "This value is good and should be named [Standardized Name] because [Your reason]".
   - "value": MUST be the exact Standardized Name from the scope.

2. "state": "decline"
   - Use when: The input is garbage, a completely different category, or invalid data.
   - "message": MUST explain why it shouldn't be ingested.
   - "value": MUST be an empty string "".

3. "state": "suggest"
   - Use when: The input is a valid item of the same category (e.g., a fruit) but is NOT in the scope.
   - "message": MUST say "This value should be added to our scope because [Your reason]".
   - "value": MUST be the cleaned name of the suggested new item.
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
                    f"Rate limited (429). Waiting {backoff_time} seconds before retry {attempt + 1}/{max_retries}...")
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

def load_scope(filepath):
    """Reads the scope.csv file and returns a list of allowed values."""
    scope_list = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if not row:
                continue
            val = row[0].strip()
            if i == 0 and val.lower() in ['scope', 'value', 'scope_value', 'allowed', 'name', 'raw_value']:
                continue
            if val:
                scope_list.append(val)
    return scope_list

def process_csv(input_filepath, scope):
    print(f"Processing data from {input_filepath}...\n")

    with open(input_filepath, mode='r', encoding='utf-8') as file:
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
    parser = argparse.ArgumentParser(description="Run LLM Data Cleaner on a specific test case folder.")
    parser.add_argument("test_folder", help="Path to the folder containing scope.csv and input_data.csv")
    args = parser.parse_args()

    folder_path = args.test_folder
    scope_file = os.path.join(folder_path, "scope.csv")
    input_file = os.path.join(folder_path, "input_data.csv")

    if not os.path.exists(scope_file):
        print(f"Error: Could not find '{scope_file}'")
        exit(1)
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'")
        exit(1)

    print(f"Loading test case from: {folder_path}")
    current_scope = load_scope(scope_file)
    print(f"Loaded Scope: {current_scope}\n" + "="*40)
    
    process_csv(input_file, current_scope)