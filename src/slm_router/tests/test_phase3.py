import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from slm_router.cloud import CloudHandler
from slm_router.router import Router


class TestPhase3CloudAndRouter(unittest.TestCase):

    def test_mock_cloud_handler_default(self):
        handler = CloudHandler(api_key="", mode="", model_name="gemini-3.6-flash")
        self.assertEqual(handler.mode, "mock")
        self.assertEqual(handler.get_api_status(), "Not Configured")

        res = handler.handle("Explain quantum entanglement.")
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "MOCK")
        self.assertEqual(res["handler"], "Cloud LLM")
        self.assertEqual(res["model"], "gemini-3.6-flash")
        self.assertIn("Cloud LLM routing selected", res["response"])

    def test_default_cloud_model_is_gemini_3_6_flash(self):
        handler = CloudHandler(api_key="", mode="mock", model_name=None)
        self.assertEqual(handler.model_name, "gemini-3.6-flash")
        res = handler.handle("Explain black holes.")
        self.assertEqual(res["model"], "gemini-3.6-flash")

    def test_mock_cloud_handler_explicit_mode(self):
        handler = CloudHandler(api_key="dummy-key", mode="mock")
        self.assertEqual(handler.mode, "mock")
        self.assertEqual(handler.get_api_status(), "Mock Mode")

        res = handler.handle("Test query")
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "MOCK")

    def test_live_cloud_handler_missing_key(self):
        handler = CloudHandler(api_key="", mode="live")
        self.assertEqual(handler.mode, "mock")
        res = handler.handle("Test query")
        self.assertTrue(res["success"])

    def test_live_cloud_handler_invalid_key_graceful_error(self):
        handler = CloudHandler(api_key="invalid-gemini-key-for-testing", mode="live")
        self.assertEqual(handler.mode, "live")
        self.assertEqual(handler.get_api_status(), "Connected")

        res = handler.handle("Test query")
        self.assertFalse(res["success"])
        self.assertEqual(res["mode"], "ERROR")

        self.assertIn("Authentication Error", res["response"])

    def test_live_cloud_handler_success_mocked_gemini(self):
        handler = CloudHandler(api_key="valid-mock-key", mode="live", model_name="gemini-3.6-flash")

        mock_response_obj = MagicMock()
        mock_response_obj.text = "Live synthesized response from Gemini."

        with patch("google.genai.Client") as mock_genai_cls:
            mock_client = MagicMock()
            mock_genai_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response_obj

            res = handler.handle("Draft a corporate strategy.")
            self.assertTrue(res["success"])
            self.assertEqual(res["mode"], "LIVE")
            self.assertEqual(res["model"], "gemini-3.6-flash")

            self.assertEqual(res["response"], "Live synthesized response from Gemini.")

    def test_router_measured_timings(self):
        mock_slm = MagicMock()
        mock_classifier = MagicMock()
        mock_decision = MagicMock()
        mock_decision.model_dump.return_value = {
            "processing": "LOCAL",
            "memory_required": False,
            "memory_request": None,
        }
        mock_classifier.classify_with_raw.return_value = (mock_decision, "raw")

        router = Router(slm=mock_slm, classifier=mock_classifier)
        res = router.route("What is 2+2?")

        self.assertEqual(res["processing"], "LOCAL")
        self.assertFalse(res["memory_required"])
        self.assertIsNone(res["memory_request"])

    def test_api_key_never_exposed_in_result(self):
        secret_key = "secret-production-gemini-key-12345"
        handler = CloudHandler(api_key=secret_key, mode="live")
        res = handler.handle("Test")
        self.assertNotIn(secret_key, str(res))



if __name__ == "__main__":
    unittest.main()
