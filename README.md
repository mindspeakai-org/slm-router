# SLM Router

A lightweight, high-performance API service that analyzes user queries and returns structured routing and memory decisions for agentic systems (such as `toy-agent`).

The router uses an on-device Small Language Model (`Qwen/Qwen2.5-1.5B-Instruct`) to make fast, deterministic routing decisions without generating conversational text, executing external commands, or retrieving memory values.

---

## Architectural Principles

1. **Pure Decision Service**: The router's sole responsibility is analyzing the user's query and outputting a structured decision. It does NOT generate the final conversational answer, call external tools, or query databases.
2. **Two Processing Targets (`LOCAL` vs `CLOUD`)**:
   - `LOCAL`: General chit-chat, math, explanations, personal questions, and device commands (hardware actions are executed locally by the client device).
   - `CLOUD`: Live/real-time external lookups (e.g., live weather, stock prices, news) and long-form complex generation (e.g., 500+ word essays, research reports, system architecture).
3. **Memory is Metadata, Not a Processing Route**:
   - `memory_required` indicates whether stored user profile information or personal facts are needed.
   - `memory_request` specifies the semantic keys required to answer the query (e.g. `["favorite_animal"]`).
   - The router returns semantic keys, never stored values (e.g., returns `"favorite_animal"`, never `"tiger"`).
   - Multiple memory requirements are resolved in a single call (e.g., `["child_name", "favorite_animal"]`).
4. **Decoupled Model Runtime**: The classifier depends on a generic model interface (`BaseModel`), allowing compatible Hugging Face open-weight models to be swapped via configuration (`LOCAL_MODEL`).

---

## Decision Contract

Every query routed through `POST /route` returns the following JSON structure:

```json
{
  "processing": "LOCAL" | "CLOUD",
  "memory_required": true | false,
  "memory_request": {
    "keys": ["key1", "key2"]
  } | null
}
```

### Constraints:
- `processing`: strictly `"LOCAL"` or `"CLOUD"`.
- `memory_required`: boolean (`true` or `false`).
- `memory_request`:
  - When `memory_required == true`: object containing `"keys"` (a non-empty array of semantic string identifiers).
  - When `memory_required == false`: strictly `null`.

---

## Requirements

- **Python**: `>= 3.11`
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/)
- **Default Reference Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Hardware Acceleration**: Automatically detects Apple Silicon (`mps`), NVIDIA GPU (`cuda`), or `cpu`.

---

## Quick Start

### 1. Installation

Clone the repository and synchronize dependencies using `uv`:

```bash
git clone <repository-url>
cd slm-router
uv sync
```

### 2. Environment Configuration

Copy the template `.env.example`:

```bash
cp .env.example .env
```

Default settings in `.env`:
```env
LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

### 3. Start the API Service

Launch the FastAPI application with Uvicorn using the source directory layout:

```bash
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8008
```

The service preloads the model into memory and listens on `http://127.0.0.1:8008`.

---

## API Reference

Interactive Swagger documentation is available at:
**[http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs)**

### 1. `GET /health`

Checks service health and reports the active local model.

```bash
curl http://127.0.0.1:8008/health
```

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

---

### 2. `POST /route`

Analyzes user query and returns the structured routing decision.

```bash
curl -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query":"What is my favorite animal?"}'
```

#### Example Scenarios:

**A. General Conversation / Joke (`LOCAL`, no memory)**:
```bash
curl -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me a joke."}'
```
```json
{
  "processing": "LOCAL",
  "memory_required": false,
  "memory_request": null
}
```

**B. Single Memory Request (`LOCAL` with memory)**:
```bash
curl -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query":"What is my favorite animal?"}'
```
```json
{
  "processing": "LOCAL",
  "memory_required": true,
  "memory_request": {
    "keys": ["favorite_animal"]
  }
}
```

**C. Multiple Memory Requests in Single Call**:
```bash
curl -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query":"What is my name and what is my favorite animal?"}'
```
```json
{
  "processing": "LOCAL",
  "memory_required": true,
  "memory_request": {
    "keys": ["child_name", "favorite_animal"]
  }
}
```

**D. Device Action / Command (processed locally by agent)**:
```bash
curl -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query":"Turn on the lights."}'
```
```json
{
  "processing": "LOCAL",
  "memory_required": false,
  "memory_request": null
}
```

**E. Live Information / Complex Task (`CLOUD`, no memory)**:
```bash
curl -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the weather today?"}'
```
```json
{
  "processing": "CLOUD",
  "memory_required": false,
  "memory_request": null
}
```

---

## External Client Integration (e.g., `toy-agent`)

A separate application can communicate with SLM-Router via HTTP:

```
toy-agent
    |
    | HTTP POST /route {"query": "..."}
    v
SLM-Router (FastAPI :8008)
    |
    v
Router.route(query)
    |
    v
ClassifierV3 (Qwen 1.5B)
    |
    v
Structured Decision JSON
```

### Python HTTP Example:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8008/route",
    json={"query": "What is my favorite animal?"},
    timeout=10.0,
)
response.raise_for_status()
decision = response.json()

if decision["processing"] == "LOCAL":
    if decision["memory_required"]:
        keys = decision["memory_request"]["keys"]
        # Client resolves memory keys locally
        pass
    else:
        # Client executes locally
        pass
else:
    # Client dispatches to Cloud LLM
    pass
```

---

## Verification & Testing

### Run All Unit Tests
```bash
PYTHONPATH=src uv run python -m unittest discover -s src/slm_router/tests -p "test_*.py"
```

### Run API Unit Tests
```bash
PYTHONPATH=src uv run python -m unittest src/slm_router/tests/test_api.py
```

### Run Router Decision Benchmark (14 Real Queries)
```bash
PYTHONPATH=src uv run python tests/test_classifier.py
```
