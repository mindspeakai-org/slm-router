"""Minimal example client demonstrating HTTP API interaction with SLM Router."""

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8008"


def make_request(method: str, path: str, payload: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"} if payload else {}
    data = json.dumps(payload).encode("utf-8") if payload else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error connecting to SLM Router API at {url}: {e}")
        print("Please ensure the FastAPI server is running with:")
        print("  uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8008")
        sys.exit(1)


def main():
    print("1. Checking API Health...")
    health = make_request("GET", "/health")
    print(f"   Status: {health.get('status')} | Active Model: {health.get('model')}\n")

    queries = [
        "Tell me a joke.",
        "What is my favorite animal?",
        "What is the weather today?",
    ]

    for q in queries:
        print(f"2. Routing Query: {q!r}...")
        decision = make_request("POST", "/route", {"query": q})
        print(f"   Decision: {json.dumps(decision, indent=2)}\n")


if __name__ == "__main__":
    main()
