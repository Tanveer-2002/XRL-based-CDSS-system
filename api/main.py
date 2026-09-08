"""FastAPI Main Server Entrypoint (Re-exported from consolidated api/app)."""
from .app import app, load_inference_artifacts

__all__ = ["app", "load_inference_artifacts"]
