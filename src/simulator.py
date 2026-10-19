import json
import time
import requests
import argparse
import os

def run_simulator(host: str, file_path: str, delay: float):
    url = f"{host}/predict"
    print(f"Traffic simulator")
    print(f"Target: {url}")
    print(f"Data file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return

    success_count = 0
    error_count = 0

    with open(file_path, 'r') as f:
        for line in f:
            payload = json.loads(line)
            
            try:
                response = requests.post(url, json=payload, timeout=5)
                
                if response.status_code == 200:
                    success_count += 1
                    pred = response.json().get('prediction', {})
                    print(f"Order {payload['order_id'][:8]} | Probability: {pred.get('probability'):.3f} | Bad Review: {pred.get('is_bad_review')}")
                else:
                    error_count += 1
                    print(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                error_count += 1
                print(f"Error of connection: {e}")

            time.sleep(delay)

    print(f"\nSimulation completed. Successful count: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Traffic Simulator")
    parser.add_argument("--host", default="http://localhost:8000", help="URL of the API")
    parser.add_argument("--file", default="models/replay.jsonl", help="Path to replay.jsonl")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    
    args = parser.parse_args()
    run_simulator(args.host, args.file, args.delay)