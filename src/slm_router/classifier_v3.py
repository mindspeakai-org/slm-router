"""Classifier V3: Structured routing and memory decision classifier."""

from typing import Optional, Tuple
from slm_router.models.base import BaseModel
from slm_router.models.transformers import TransformersRuntime
from slm_router.schemas import RouteDecision, parse_decision

VALID_LABELS = {"LOCAL", "CLOUD"}

CLASSIFIER_V3_SYSTEM_PROMPT = """You are the query routing and memory classifier for an AI agent.

CRITICAL RULE:
This is a ROUTING CLASSIFICATION TASK ONLY.
NEVER answer, explain, calculate, summarize, perform, or fulfill the user's request.
Even if the request asks 'Explain...', 'Tell me...', 'What is...', 'Write...', or 'How do I...', DO NOT ANSWER IT.
You are NOT an answering assistant. You are ONLY a query routing classifier.
Output ONLY the raw JSON object starting with { and ending with }. No markdown code fences, no conversational text.

Required JSON Structure:
{
  "processing": "LOCAL" or "CLOUD",
  "memory_required": true or false,
  "memory_request": {
    "keys": ["key1", "key2"]
  } or null
}

Field Definitions and Rules:

1. "processing" ("LOCAL" or "CLOUD"):
   - "LOCAL":
     * General conversation, chit-chat, greetings, humor (e.g., "Hello", "Tell me a joke", "How are you?").
     * Direct factual questions, definitions, basic calculations, short explanations (e.g., "What is 2 + 2?", "Explain photosynthesis").
     * Device, hardware, appliance, or application commands (e.g., "Turn on the light", "Turn off the fan", "Open the door", "Play music", "Launch browser"). These are executed locally by the agent.
     * Personal queries that can be answered locally once user memory is provided (e.g., "What is my favorite animal?", "What is my name?").
   - "CLOUD":
     * Live, external, or real-time information requiring web access (e.g., "What is the weather today?", "Latest news", "Stock prices").
     * Tasks exceeding local budget: long-form writing (500+ word essays, extensive stories, research reports), complex multi-day planning, or deep technical architecture.

2. "memory_required" (true or false):
   - true: Answering the query requires personal context, saved user preferences, biographical details, or private stored facts about the user/child (e.g., favorite animal, name, age, birthday, friends, pets, school, hobbies).
   - false: The query does not require stored personal user memory.
   
   CRITICAL DISTINCTION FOR MEMORY:
   "memory_required" applies EXCLUSIVELY to personal profile information, preferences, and personal memories about the user/child (e.g. user's name, favorite animal, age).
   General world knowledge, public facts, academic subjects, and technical concepts (e.g. climate change, distributed systems, quantum computing, dragons, photosynthesis) are NOT user memories.
   NEVER set memory_required=true for general topics. For general topics, essays, research, or world knowledge, memory_required MUST be false and memory_request MUST be null.

3. "memory_request":
   - When memory_required is true: MUST be an object {"keys": ["..."]} containing a non-empty array of semantic identifier keys for the required personal information (e.g., "favorite_animal", "child_name", "pet_name", "birthday").
     CRITICAL: Keys must identify the needed information attribute, NEVER the actual stored value (e.g., "favorite_animal", NEVER "tiger").
     If multiple pieces of memory are needed, resolve them all in one call by listing all keys in the array (e.g., ["child_name", "favorite_animal"]).
   - When memory_required is false: MUST be null.

Examples:

User: Request to classify: What is my favorite animal?
{"processing": "LOCAL", "memory_required": true, "memory_request": {"keys": ["favorite_animal"]}}

User: Request to classify: What is my name?
{"processing": "LOCAL", "memory_required": true, "memory_request": {"keys": ["child_name"]}}

User: Request to classify: Explain photosynthesis in simple terms.
{"processing": "LOCAL", "memory_required": false, "memory_request": null}

User: Request to classify: Tell me a joke.
{"processing": "LOCAL", "memory_required": false, "memory_request": null}

User: Request to classify: What is the weather today?
{"processing": "CLOUD", "memory_required": false, "memory_request": null}

User: Request to classify: What is my name and what is my favorite animal?
{"processing": "LOCAL", "memory_required": true, "memory_request": {"keys": ["child_name", "favorite_animal"]}}

User: Request to classify: Turn on the lights.
{"processing": "LOCAL", "memory_required": false, "memory_request": null}

User: Request to classify: What is 2 + 2?
{"processing": "LOCAL", "memory_required": false, "memory_request": null}

User: Request to classify: Tell me a 500-word story about a dragon.
{"processing": "CLOUD", "memory_required": false, "memory_request": null}

User: Request to classify: Write a 3000-word essay about climate change.
{"processing": "CLOUD", "memory_required": false, "memory_request": null}

User: Request to classify: Develop a detailed production-ready distributed system architecture.
{"processing": "CLOUD", "memory_required": false, "memory_request": null}

User: Request to classify: Write a 500-word story about my favorite animal.
{"processing": "CLOUD", "memory_required": true, "memory_request": {"keys": ["favorite_animal"]}}
"""


def extract_label(raw_output: str) -> str:
    """Extract processing label from raw output for backward compatibility."""
    if not raw_output:
        return "UNKNOWN"
    try:
        decision = parse_decision(raw_output)
        return decision.processing
    except Exception:
        normalized = raw_output.strip().upper().rstrip(".!,:;")
        if normalized in VALID_LABELS:
            return normalized
        if "LOCAL" in normalized and "CLOUD" not in normalized:
            return "LOCAL"
        if "CLOUD" in normalized and "LOCAL" not in normalized:
            return "CLOUD"
        return "UNKNOWN"


class ClassifierV3:
    def __init__(self, model: Optional[BaseModel] = None):
        self.model = model if model is not None else TransformersRuntime()
        self.slm = self.model  # Backward-compatibility alias
        self.last_raw_output = ""

    def classify(self, query: str) -> RouteDecision:
        decision, _ = self.classify_with_raw(query)
        return decision

    def classify_with_raw(self, query: str) -> Tuple[RouteDecision, str]:
        messages = [
            {"role": "system", "content": CLASSIFIER_V3_SYSTEM_PROMPT},
            {"role": "user", "content": f"Request to classify: {query}"},
        ]

        raw_output = self.model.generate(
            messages=messages,
            max_new_tokens=100,
            do_sample=False,
        )

        self.last_raw_output = raw_output
        decision = parse_decision(raw_output)
        return decision, raw_output
