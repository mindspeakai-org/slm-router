"""Unit tests for FastAPI transport layer."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure src is in sys.path
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi.testclient import TestClient
from slm_router.api import app, set_router
from slm_router.schemas import InvalidModelOutputError, RouteDecision


class TestAPI(unittest.TestCase):
    """Test FastAPI endpoints using mocked Router to prevent model loading."""

    def setUp(self):
        self.mock_router = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.model_name = "MockedModel/Mock-1B"
        self.mock_router.model = self.mock_model

        self.mock_router.route.return_value = {
            "processing": "LOCAL",
            "memory_required": False,
            "memory_request": None,
        }

        # Inject mock router before TestClient runs lifespan
        set_router(self.mock_router)
        self.client = TestClient(app)

    def tearDown(self):
        set_router(None)

    def test_health_endpoint(self):
        """GET /health returns 200 with healthy status and active model name."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "MockedModel/Mock-1B")
        self.assertIsInstance(json.dumps(data), str)

    def test_route_valid_local_query(self):
        """POST /route returns 200 with exact RouteDecision schema."""
        response = self.client.post("/route", json={"query": "Tell me a joke."})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["processing"], "LOCAL")
        self.assertFalse(data["memory_required"])
        self.assertIsNone(data["memory_request"])
        self.mock_router.route.assert_called_once_with("Tell me a joke.")

    def test_route_valid_memory_query(self):
        """POST /route with memory requirement returns memory_request object."""
        self.mock_router.route.return_value = {
            "processing": "LOCAL",
            "memory_required": True,
            "memory_request": {"keys": ["favorite_animal"]},
        }
        response = self.client.post("/route", json={"query": "What is my favorite animal?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["processing"], "LOCAL")
        self.assertTrue(data["memory_required"])
        self.assertEqual(data["memory_request"]["keys"], ["favorite_animal"])

    def test_route_rejects_missing_query(self):
        """POST /route returns 422 when query field is missing."""
        response = self.client.post("/route", json={})
        self.assertEqual(response.status_code, 422)

    def test_route_rejects_empty_query(self):
        """POST /route returns 422 when query is empty string."""
        response = self.client.post("/route", json={"query": ""})
        self.assertEqual(response.status_code, 422)

    def test_route_handles_invalid_model_output_with_502(self):
        """POST /route returns 502 Bad Gateway when model output is invalid."""
        self.mock_router.route.side_effect = InvalidModelOutputError("Malformed model JSON")
        response = self.client.post("/route", json={"query": "Hello"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("Model produced invalid routing output", response.json()["detail"])

    def test_classify_deprecated_endpoint_works(self):
        """POST /classify returns 200 matching RouteDecision for backwards compatibility."""
        response = self.client.post("/classify", json={"query": "Tell me a joke."})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["processing"], "LOCAL")
        self.assertFalse(data["memory_required"])


if __name__ == "__main__":
    unittest.main()
