# SLM Router — HTTP API Reference

The SLM Router exposes a lightweight FastAPI HTTP interface that wraps the underlying `Router` and `ClassifierV3` pipeline.

The API runs by default on `http://127.0.0.1:8008`.

---

## Server Lifecycle

- The active Hugging Face model (`Qwen/Qwen2.5-1.5B-Instruct`) and `Router` singleton are initialized once during FastAPI application startup (`lifespan`) and kept in memory.
- Subsequent requests reuse the resident model weights with zero cold-start overhead.
- No model reloading occurs per request.

To start the server:
```bash
uv run uvicorn --app-dir src slm_router.api:app --host 127.0.0.1 --port 8008
```

---

## Endpoints Overview

| Method | Path | Request Body | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | Service liveness probe & active model identifier |
| `POST` | `/route` | `{"query": "..."}` | Query analysis returning structured processing and memory decision |
| `POST` | `/classify` | `{"query": "..."}` | *(Deprecated)* Alias to `/route` returning the structured decision |

---

## 1. Health Probe (`GET /health`)

### Purpose
Confirms that the FastAPI service is active and dynamically returns the identifier of the configured local model currently loaded in memory.

### Request
- **Method**: `GET`
- **Path**: `/health`
- **Headers**: None required
- **Body**: None

### Response (`200 OK`)
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

### Example Curl
```bash
curl -s http://127.0.0.1:8008/health
```

---

## 2. Route Query (`POST /route`)

### Purpose
Analyzes the user's query and returns a structured routing decision.

The router's sole responsibility is returning the decision:
- `processing`: target processing tier (`"LOCAL"` or `"CLOUD"`)
- `memory_required`: boolean (`true` or `false`)
- `memory_request`: required memory keys (`{"keys": [...]}`) if `memory_required` is true; `null` otherwise.

The router does NOT generate final answers, execute external tools, or retrieve memory values.

### Request
- **Method**: `POST`
- **Path**: `/route`
- **Content-Type**: `application/json`

**Body:**
```json
{
  "query": "What is my favorite animal?"
}
```

### Response Schema (`200 OK`)
```json
{
  "processing": "LOCAL",
  "memory_required": true,
  "memory_request": {
    "keys": ["favorite_animal"]
  }
}
```

### Representative Examples

#### A. General Factual or Conversational (`LOCAL`, no memory)
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me a joke."}'
```
**Response:**
```json
{
  "processing": "LOCAL",
  "memory_required": false,
  "memory_request": null
}
```

#### B. Personal Memory Query (`LOCAL` with memory)
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my favorite animal?"}'
```
**Response:**
```json
{
  "processing": "LOCAL",
  "memory_required": true,
  "memory_request": {
    "keys": ["favorite_animal"]
  }
}
```

#### C. Multiple Memory Keys in Single Call
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my name and what is my favorite animal?"}'
```
**Response:**
```json
{
  "processing": "LOCAL",
  "memory_required": true,
  "memory_request": {
    "keys": ["child_name", "favorite_animal"]
  }
}
```

#### D. Device Action / Command Query
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Turn on the light."}'
```
**Response:**
```json
{
  "processing": "LOCAL",
  "memory_required": false,
  "memory_request": null
}
```

#### E. Live/Real-time or Complex Task (`CLOUD`, no memory)
**Request:**
```bash
curl -s -X POST http://127.0.0.1:8008/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather today?"}'
```
**Response:**
```json
{
  "processing": "CLOUD",
  "memory_required": false,
  "memory_request": null
}
```

---

## 3. Error Responses

### Missing Query (`422 Unprocessable Entity`)
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required"
    }
  ]
}
```

### Empty Query (`422 Unprocessable Entity`)
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character"
    }
  ]
}
```

### Invalid Model Output (`502 Bad Gateway`)
Returned if the underlying language model generates malformed JSON or an invalid decision structure.
```json
{
  "detail": "Model produced invalid routing output: ..."
}
```
