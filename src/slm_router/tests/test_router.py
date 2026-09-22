"""Unit tests for Router, ClassifierV3, and decision schemas."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure src is in sys.path
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from slm_router.models.base import BaseModel
from slm_router.classifier_v3 import ClassifierV3
from slm_router.router import Router
from slm_router.schemas import (
    InvalidModelOutputError,
    MemoryRequest,
    RouteDecision,
    parse_decision,
)


class DummyTestModel(BaseModel):
    def __init__(self, response: str = ""):
        self.response = response
        self.last_generate_kwargs = {}

    def generate(self, prompt=None, messages=None, max_new_tokens=100, do_sample=False, **kwargs):
        self.last_generate_kwargs = {
            "prompt": prompt,
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            **kwargs,
        }
        return self.response


class TestRouter(unittest.TestCase):
    """Test suite for Router decision contract and ClassifierV3 behavior."""

    def setUp(self):
        self.mock_model = DummyTestModel()
        self.classifier = ClassifierV3(model=self.mock_model)
        self.router = Router(model=self.mock_model, classifier=self.classifier)

    # 1. Normal LOCAL query
    def test_normal_local_query(self):
        """Query like 'Tell me a joke' produces LOCAL with no memory."""
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": False,
            "memory_request": None,
        })
        result = self.router.route("Tell me a joke.")
        self.assertEqual(result["processing"], "LOCAL")
        self.assertFalse(result["memory_required"])
        self.assertIsNone(result["memory_request"])

    # 2. Normal CLOUD query
    def test_normal_cloud_query(self):
        """Query like 'What is the weather today?' produces CLOUD with no memory."""
        self.mock_model.response = json.dumps({
            "processing": "CLOUD",
            "memory_required": False,
            "memory_request": None,
        })
        result = self.router.route("What is the weather today?")
        self.assertEqual(result["processing"], "CLOUD")
        self.assertFalse(result["memory_required"])
        self.assertIsNone(result["memory_request"])

    # 3. Memory-required LOCAL query
    def test_memory_required_local_query(self):
        """Query like 'What is my favorite animal?' returns LOCAL with memory request."""
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": {"keys": ["favorite_animal"]},
        })
        result = self.router.route("What is my favorite animal?")
        self.assertEqual(result["processing"], "LOCAL")
        self.assertTrue(result["memory_required"])
        self.assertIsNotNone(result["memory_request"])
        self.assertEqual(result["memory_request"]["keys"], ["favorite_animal"])

    # 4. Memory-required CLOUD query
    def test_memory_required_cloud_query(self):
        """Query requiring cloud generation and user memory returns CLOUD with memory request."""
        self.mock_model.response = json.dumps({
            "processing": "CLOUD",
            "memory_required": True,
            "memory_request": {"keys": ["favorite_animal"]},
        })
        result = self.router.route("Write a 500-word story about my favorite animal.")
        self.assertEqual(result["processing"], "CLOUD")
        self.assertTrue(result["memory_required"])
        self.assertEqual(result["memory_request"]["keys"], ["favorite_animal"])

    # 5. Multiple memory keys resolved in one classification call
    def test_multiple_memory_keys_resolved_in_single_call(self):
        """Query needing multiple facts returns all keys in a single response."""
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": {"keys": ["child_name", "favorite_animal"]},
        })
        result = self.router.route("What is my name and what is my favorite animal?")
        self.assertEqual(result["processing"], "LOCAL")
        self.assertTrue(result["memory_required"])
        self.assertEqual(result["memory_request"]["keys"], ["child_name", "favorite_animal"])

    # 6. Command-like query being classified as LOCAL
    def test_command_like_query_classified_as_local(self):
        """Device commands like 'Turn on the lights' are routed to LOCAL with no memory."""
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": False,
            "memory_request": None,
        })
        result = self.router.route("Turn on the lights.")
        self.assertEqual(result["processing"], "LOCAL")
        self.assertFalse(result["memory_required"])
        self.assertIsNone(result["memory_request"])

    # 7. No memory request => memory_request is null
    def test_no_memory_request_is_null(self):
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": False,
            "memory_request": None,
        })
        result = self.router.route("What is 2 + 2?")
        self.assertIsNone(result["memory_request"])

    # 8. memory_required=False => memory_request must be null (rejects inconsistent output)
    def test_memory_required_false_with_keys_is_invalid(self):
        """If memory_required is false but keys are provided, it must be rejected."""
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": False,
            "memory_request": {"keys": ["favorite_animal"]},
        })
        with self.assertRaises(InvalidModelOutputError):
            self.router.route("Tell me a joke.")

    # 9. Invalid/malformed model output (not valid JSON)
    def test_malformed_model_output_raises_error(self):
        """Non-JSON model output must raise InvalidModelOutputError and never silently pass."""
        self.mock_model.response = "I am an AI assistant and I think this is LOCAL."
        with self.assertRaises(InvalidModelOutputError):
            self.router.route("What is 2 + 2?")

    # 10. Missing required fields in model output
    def test_missing_required_fields_raises_error(self):
        """Output missing required fields (e.g. missing memory_required) must be rejected."""
        self.mock_model.response = json.dumps({"processing": "LOCAL"})
        with self.assertRaises(InvalidModelOutputError):
            self.router.route("What is 2 + 2?")

    # 11. Invalid processing value (e.g. obsolete 'COMMAND' or 'UNKNOWN')
    def test_invalid_processing_value_raises_error(self):
        """Obsolete processing values like COMMAND must not be accepted."""
        self.mock_model.response = json.dumps({
            "processing": "COMMAND",
            "memory_required": False,
            "memory_request": None,
        })
        with self.assertRaises(InvalidModelOutputError):
            self.router.route("Turn on the light.")

    # 12. Invalid memory_request structure (memory_required=true but null keys)
    def test_memory_required_true_with_null_memory_request_raises_error(self):
        """memory_required=True with null memory_request must be rejected."""
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": None,
        })
        with self.assertRaises(InvalidModelOutputError):
            self.router.route("What is my name?")

    def test_memory_required_true_with_empty_keys_raises_error(self):
        """memory_required=True with empty keys list must be rejected."""
        self.mock_model.response = json.dumps({
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": {"keys": []},
        })
        with self.assertRaises(InvalidModelOutputError):
            self.router.route("What is my name?")

    # 13. Empty query rejected
    def test_empty_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.router.route("   ")

    # 14. Markdown code block stripped cleanly
    def test_markdown_wrapped_json_parsed_successfully(self):
        self.mock_model.response = "```json\n{\"processing\": \"LOCAL\", \"memory_required\": false, \"memory_request\": null}\n```"
        result = self.router.route("Hello there")
        self.assertEqual(result["processing"], "LOCAL")

    # 15. Security test: ensure no subprocess or shell calls exist in router
    def test_no_arbitrary_shell_execution_possible(self):
        router_path = Path(__file__).resolve().parent.parent / "router.py"
        with open(router_path, "r", encoding="utf-8") as f:
            content = f.read()

        dangerous_patterns = [
            "import subprocess",
            "from subprocess",
            "os.system",
            "os.popen",
            "shell=True",
            "exec(",
            "eval(",
        ]
        for pattern in dangerous_patterns:
            self.assertNotIn(pattern, content)


if __name__ == "__main__":
    unittest.main()
