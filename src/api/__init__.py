"""
API module for Real-Time Recommendation Engine
FastAPI-based service with sub-100ms latency
"""

from .recommendation_api import app

__all__ = ["app"]
