"""FastAPI transport layer for SLM Router."""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from slm_router.router import Router
from slm_router.schemas import RouteDecision, InvalidModelOutputError

_router_instance: Optional[Router] = None


def get_router() -> Router:
    """Retrieve or initialize the singleton Router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = Router()
    return _router_instance


def set_router(router: Optional[Router]) -> None:
    """Set or reset the global Router instance (useful for testing/mocking)."""
    global _router_instance
    _router_instance = router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialize the model/router on application startup."""
    get_router()
    yield


app = FastAPI(
    title="SLM Router API",
    description="Lightweight HTTP service for query classification and memory requirement planning",
    version="0.2.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to analyze and route")


class HealthResponse(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
def health(router: Router = Depends(get_router)) -> HealthResponse:
    """Check API health and return the active local model name."""
    model_name = getattr(router.model, "model_name", "unknown")
    return HealthResponse(status="healthy", model=model_name)


@app.post("/route", response_model=RouteDecision)
def route(request: QueryRequest, router: Router = Depends(get_router)) -> Dict[str, Any]:
    """Analyze query and return structured routing and memory decision."""
    try:
        return router.route(request.query)
    except InvalidModelOutputError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model produced invalid routing output: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/classify", response_model=RouteDecision, deprecated=True)
def classify_deprecated(request: QueryRequest, router: Router = Depends(get_router)) -> Dict[str, Any]:
    """Deprecated: Use /route instead. Returns the structured routing decision."""
    try:
        return router.route(request.query)
    except InvalidModelOutputError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model produced invalid routing output: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
