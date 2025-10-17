"""
Analytics Router - Basic placeholder
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/overview")
async def get_analytics_overview():
    """Get analytics overview - placeholder"""
    return {"conversations_today": 0, "languages_used": ["en", "hi"]}
