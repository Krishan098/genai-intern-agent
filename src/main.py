from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging


from src.routers.blog_routes import router as blog_router
from src.utils.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)




app = FastAPI(
    title="GenAI Intern Agent",
    description="Agentic blog support system for keyword recommendations and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
def root():
    """Root endpoint"""
    return {
        "message": "GenAI Intern Agent API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
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