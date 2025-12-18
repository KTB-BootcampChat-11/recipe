from .analyze import router as analyze_router
from .test import router as test_router
from .health import router as health_router
from .chat import router as chat_router

__all__ = ["analyze_router", "test_router", "health_router", "chat_router"]
