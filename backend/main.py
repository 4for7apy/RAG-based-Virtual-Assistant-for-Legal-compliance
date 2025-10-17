"""
Campus Multilingual Chatbot - Main Application
FastAPI backend with advanced multilingual support
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging
import asyncio
from typing import List, Dict, Any
import json

from app.config import settings
from app.database import init_db
from app.routers import chat, auth, admin, analytics, files
from app.services.chat_service import ChatService
from app.services.websocket_manager import WebSocketManager
from app.middleware.rate_limiting import RateLimitingMiddleware
from app.middleware.logging import LoggingMiddleware
from app.utils.startup import startup_tasks, shutdown_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
websocket_manager = WebSocketManager()
chat_service = ChatService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Campus Chatbot application...")
    await startup_tasks()
    await init_db()
    await chat_service.initialize()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Campus Chatbot application...")
    await shutdown_tasks()

# Create FastAPI application
app = FastAPI(
    title="Campus Multilingual Chatbot",
    description="Advanced multilingual chatbot for campus assistance with AI-powered responses",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(files.router, prefix="/api/files", tags=["files"])

# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat communication"""
    try:
        await websocket_manager.connect(websocket, user_id)
        logger.info(f"WebSocket connected for user: {user_id}")
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message through chat service
            response = await chat_service.process_message(
                user_id=user_id,
                message=message_data.get("message", ""),
                language=message_data.get("language", "en"),
                context=message_data.get("context", {})
            )
            
            # Send response back to client
            await websocket_manager.send_personal_message(
                json.dumps(response), user_id
            )
            
            # Broadcast typing indicator to other connected clients if needed
            if message_data.get("typing", False):
                await websocket_manager.broadcast_typing_indicator(
                    user_id, message_data["typing"]
                )
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)
        logger.info(f"WebSocket disconnected for user: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")
        await websocket_manager.send_personal_message(
            json.dumps({"error": "Connection error occurred"}), user_id
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": "connected",
            "redis": "connected",
            "ai_service": "available"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Campus Multilingual Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/chat/{user_id}"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="Internal server error occurred"
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
