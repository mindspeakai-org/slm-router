# SLM Router — Developer Integration Guide

This guide explains how external services, edge applications, and Python codebases (such as `toy-agent`) can integrate with SLM Router.

---

## 1. High-Level Integration Flow

External applications submit raw natural language requests to SLM Router over HTTP or directly in Python. The router analyzes the request and returns a structured decision (`processing: "LOCAL" | "CLOUD"`, `memory_required`, and `memory_request`).

```
External Application (e.g. toy-agent)
        ↓
HTTP POST /route {"query": "..."}
        ↓
    SLM Router
        ↓
ClassifierV3 (Qwen 1.5B)
        ↓
Structured Decision:
{
  "processing": "LOCAL" | "CLOUD",
  "memory_required": bool,
  "memory_request": {"keys": [...]} | null
}
```

The router does NOT generate answers, execute commands, or retrieve memory. The client application executes the decision.

---

## 2. HTTP Integration (Recommended for `toy-agent`)

### Starting the Service
```bash
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8008
```

### Python HTTP Client Example:
```python
import requests

response = requests.post(
    "http://127.0.0.1:8008/route",
    json={"query": "What is my favorite animal?"},
    timeout=10.0,
)
response.raise_for_status()
decision = response.json()

# Example structure returned:
# {
#   "processing": "LOCAL",
#   "memory_required": True,
#   "memory_request": {"keys": ["favorite_animal"]}
# }

if decision["processing"] == "LOCAL":
    if decision["memory_required"]:
        required_keys = decision["memory_request"]["keys"]
        # Retrieve personal facts for required_keys from local storage
    # Execute locally on agent
else:
    # Offload to cloud LLM
    pass
```

---

## 3. In-Process Python Usage

For Python applications running on the same host wishing to call the router directly:

```python
from slm_router.router import Router

# Initialize the router (loads model weights into memory once)
router = Router()

# Route a query
decision = router.route("What is my name and favorite animal?")

print("Processing:", decision["processing"])
print("Memory Required:", decision["memory_required"])
print("Memory Keys:", decision["memory_request"]["keys"])
```

---

## 4. Configuration Reference (`LOCAL_MODEL`)

You can change the active local causal language model without modifying application code:

1. In your `.env` file:
   ```env
   LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
   ```
2. Or via environment variable before launching the server or Python script:
   ```bash
   LOCAL_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8008
   ```

If unset, the system defaults to `Qwen/Qwen2.5-1.5B-Instruct`.
