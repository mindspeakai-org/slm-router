"""Router component: analyzes query and returns structured routing/memory decision."""

from typing import Any, Dict, Optional

from slm_router.models.base import BaseModel
from slm_router.models.transformers import TransformersRuntime
from slm_router.classifier_v3 import ClassifierV3
from slm_router.schemas import RouteDecision


class Router:
    """Orchestrates query classification and memory requirement planning."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        classifier: Optional[ClassifierV3] = None,
        command_executor: Optional[Any] = None,
        cloud_handler: Optional[Any] = None,
        slm: Optional[BaseModel] = None,
    ):
        # Allow either model or slm parameter for backwards compatibility
        active_model = model if model is not None else slm
        self.model = active_model if active_model is not None else TransformersRuntime()
        self.slm = self.model  # Backward compatibility alias

        self.classifier = classifier if classifier is not None else ClassifierV3(self.model)
        self.commands = command_executor  # Preserved for backward-compatible initialization
        self.cloud = cloud_handler        # Preserved for backward-compatible initialization

    def route(self, query: str) -> Dict[str, Any]:
        """Analyze user query and return a structured decision dictionary.

        The router's sole responsibility is returning the structured decision:
        - processing destination ('LOCAL' or 'CLOUD')
        - memory_required (boolean)
        - memory_request ({'keys': [...]} or None)

        The router does NOT generate final conversational answers, retrieve memory,
        or execute external actions.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Empty query provided. Query string cannot be empty.")

        decision, _ = self.classifier.classify_with_raw(cleaned_query)
        return decision.model_dump()
