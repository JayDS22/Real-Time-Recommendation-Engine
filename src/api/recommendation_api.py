"""
Real-Time Recommendation API
High-performance FastAPI service for serving recommendations with <100ms latency
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import structlog
import yaml
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from ..models.recommendation_engine import RecommendationEngine
from ..utils.cache import CacheManager
from ..utils.metrics import MetricsCollector

# Configure structured logging
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request latency')
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')
RECOMMENDATION_QUALITY = Histogram('recommendation_quality', 'Recommendation quality metrics', ['metric'])

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = Field(default=10, ge=1, le=50)
    exclude_seen: bool = True
    algorithm: str = Field(default="hybrid", regex="^(svd|nmf|hybrid)$")
    
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    algorithm_used: str
    response_time_ms: float
    cache_hit: bool

class UserInteraction(BaseModel):
    user_id: int
    item_id: int
    rating: float = Field(ge=1, le=5)
    timestamp: Optional[str] = None
    interaction_type: str = Field(default="rating")

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    uptime_seconds: float
    active_models: List[str]

# Global variables
app_start_time = time.time()
recommendation_engine = None
cache_manager = None
metrics_collector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global recommendation_engine, cache_manager, metrics_collector
    
    logger.info("Starting Recommendation API...")
    
    # Initialize components
    recommendation_engine = RecommendationEngine(config)
    cache_manager = CacheManager(config['database']['redis'])
    metrics_collector = MetricsCollector()
    
    # Load pre-trained models
    await recommendation_engine.load_models()
    logger.info("Models loaded successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Recommendation API...")
    await cache_manager.close()

app = FastAPI(
    title="Real-Time Recommendation Engine",
    description="High-performance recommendation system with <100ms latency",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_cache_manager() -> CacheManager:
    return cache_manager

async def get_recommendation_engine() -> RecommendationEngine:
    return recommendation_engine

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    REQUEST_LATENCY.observe(process_time)
    
    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app_start_time
    
    # Check model availability
    active_models = []
    if recommendation_engine:
        active_models = await recommendation_engine.get_active_models()
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        active_models=active_models
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    cache: CacheManager = Depends(get_cache_manager),
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get personalized recommendations for a user"""
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"rec:{request.user_id}:{request.algorithm}:{request.num_recommendations}"
        cached_result = await cache.get(cache_key)
        
        if cached_result and not request.exclude_seen:
            response_time = (time.time() - start_time) * 1000
            logger.info("Cache hit", user_id=request.user_id, response_time_ms=response_time)
            
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=cached_result,
                algorithm_used=request.algorithm,
                response_time_ms=response_time,
                cache_hit=True
            )
        
        # Generate recommendations
        recommendations = await engine.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            algorithm=request.algorithm,
            exclude_seen=request.exclude_seen
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # Ensure sub-100ms latency requirement
        if response_time > 100:
            logger.warning(
                "High latency detected",
                user_id=request.user_id,
                response_time_ms=response_time,
                algorithm=request.algorithm
            )
        
        # Cache the result
        background_tasks.add_task(
            cache.set,
            cache_key,
            recommendations,
            config['api']['cache_ttl']
        )
        
        # Update metrics
        ACTIVE_USERS.inc()
        background_tasks.add_task(ACTIVE_USERS.dec)
        
        logger.info(
            "Recommendations generated",
            user_id=request.user_id,
            num_recommendations=len(recommendations),
            algorithm=request.algorithm,
            response_time_ms=response_time
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            algorithm_used=request.algorithm,
            response_time_ms=response_time,
            cache_hit=False
        )
        
    except Exception as e:
        logger.error("Recommendation generation failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/interactions")
async def record_interaction(
    interaction: UserInteraction,
    background_tasks: BackgroundTasks,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Record user interaction for model updates"""
    try:
        # Process interaction asynchronously
        background_tasks.add_task(
            engine.record_interaction,
            interaction.dict()
        )
        
        logger.info(
            "Interaction recorded",
            user_id=interaction.user_id,
            item_id=interaction.item_id,
            rating=interaction.rating
        )
        
        return {"status": "success", "message": "Interaction recorded"}
        
    except Exception as e:
        logger.error("Failed to record interaction", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")

@app.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: int,
    num_recommendations: int = 10,
    algorithm: str = "hybrid",
    cache: CacheManager = Depends(get_cache_manager),
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Simplified endpoint for getting user recommendations"""
    request = RecommendationRequest(
        user_id=user_id,
        num_recommendations=num_recommendations,
        algorithm=algorithm
    )
    
    background_tasks = BackgroundTasks()
    return await get_recommendations(request, background_tasks, cache, engine)

@app.get("/stats")
async def get_system_stats(engine: RecommendationEngine = Depends(get_recommendation_engine)):
    """Get system statistics and model performance"""
    try:
        stats = await engine.get_model_stats()
        return {
            "status": "success",
            "stats": stats,
            "uptime_seconds": time.time() - app_start_time
        }
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/models/retrain")
async def trigger_model_retrain(
    background_tasks: BackgroundTasks,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Trigger model retraining"""
    try:
        background_tasks.add_task(engine.retrain_models)
        logger.info("Model retraining triggered")
        return {"status": "success", "message": "Model retraining started"}
    except Exception as e:
        logger.error("Failed to trigger retraining", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "recommendation_api:app",
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers'],
        reload=False,
        access_log=True
    )
