"""
Admin Router - Basic placeholder
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/stats")
async def get_admin_stats():
    """Get admin statistics - placeholder"""
    return {"total_conversations": 0, "active_users": 0}
