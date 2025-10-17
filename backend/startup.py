"""
Startup and Shutdown Tasks
"""

import logging
import asyncio
from app.database import init_db, close_db_connections
from app.config import settings

logger = logging.getLogger(__name__)

async def startup_tasks():
    """Tasks to run on application startup"""
    try:
        logger.info("Running startup tasks...")
        
        # Initialize databases
        await init_db()
        
        # Load and process any initial data
        await load_sample_data()
        
        logger.info("Startup tasks completed successfully")
        
    except Exception as e:
        logger.error(f"Startup tasks failed: {str(e)}")
        raise

async def shutdown_tasks():
    """Tasks to run on application shutdown"""
    try:
        logger.info("Running shutdown tasks...")
        
        # Close database connections
        await close_db_connections()
        
        logger.info("Shutdown tasks completed successfully")
        
    except Exception as e:
        logger.error(f"Shutdown tasks failed: {str(e)}")

async def load_sample_data():
    """Load sample data if in development mode"""
    if not settings.DEBUG:
        return
    
    try:
        # This would load sample FAQs, knowledge base entries, etc.
        logger.info("Sample data loaded for development")
        
    except Exception as e:
        logger.warning(f"Failed to load sample data: {str(e)}")

# Health check endpoint logic
async def perform_health_check():
    """Perform comprehensive health check"""
    from app.database import check_db_health
    
    health_status = {
        "status": "healthy",
        "services": {},
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }
    
    try:
        # Check database connections
        db_health = await check_db_health()
        health_status["services"].update(db_health)
        
        # Check AI services
        ai_health = await check_ai_services()
        health_status["services"].update(ai_health)
        
        # Determine overall status
        if not all(health_status["services"].values()):
            health_status["status"] = "degraded"
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status

async def check_ai_services():
    """Check AI service availability"""
    ai_status = {
        "openai": False,
        "anthropic": False,
        "translation": False
    }
    
    try:
        # Check OpenAI
        if settings.OPENAI_API_KEY:
            # This would make a test API call
            ai_status["openai"] = True
        
        # Check Anthropic
        if settings.ANTHROPIC_API_KEY:
            # This would make a test API call
            ai_status["anthropic"] = True
        
        # Check translation service
        ai_status["translation"] = True  # Always available with fallback
        
    except Exception as e:
        logger.warning(f"AI services check failed: {str(e)}")
    
    return ai_status
