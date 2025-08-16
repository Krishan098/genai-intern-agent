from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from src.routers.blog_routes import router as blog_router
from src.utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting GenAI Intern Agent API")
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise RuntimeError("OpenAI API key is required")
    
    logger.info("API initialization complete")
    yield
    logger.info("Shutting down GenAI Intern Agent API")


app = FastAPI(
    title="GenAI Intern Agent",
    description="Agentic blog support system for keyword recommendations and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(blog_router, prefix="/api", tags=["blog"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GenAI Intern Agent API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )