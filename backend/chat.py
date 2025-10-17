"""
Chat Router
REST API endpoints for chat functionality
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging

from app.models.conversation import ChatRequest, ChatResponse, Message
from app.services.chat_service import ChatService
from app.utils.auth import get_current_user
from app.utils.rate_limiter import rate_limit

logger = logging.getLogger(__name__)
router = APIRouter()

# Global chat service instance
chat_service = ChatService()

@router.post("/message", response_model=ChatResponse)
@rate_limit(requests_per_minute=30)
async def send_message(
    request: ChatRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Send a message to the chatbot and get AI response"""
    
    try:
        user_id = current_user.get("user_id") if current_user else "anonymous"
        
        response = await chat_service.process_message(
            user_id=user_id,
            message=request.message,
            language=request.language,
            context=request.context
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process message"
        )

@router.post("/message/stream")
@rate_limit(requests_per_minute=20)
async def send_message_stream(
    request: ChatRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Send a message and get streaming AI response"""
    
    async def generate_stream():
        try:
            user_id = current_user.get("user_id") if current_user else "anonymous"
            
            # This would implement streaming response from AI models
            response = await chat_service.process_message(
                user_id=user_id,
                message=request.message,
                language=request.language,
                context=request.context
            )
            
            # Simulate streaming by yielding chunks
            words = response["response"].split()
            for i, word in enumerate(words):
                chunk = {
                    "content": word + " ",
                    "is_complete": i == len(words) - 1,
                    "metadata": response.get("metadata", {})
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
                
        except Exception as e:
            error_chunk = {
                "error": str(e),
                "is_complete": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )

@router.get("/conversation/{user_id}")
async def get_conversation_history(
    user_id: str,
    limit: int = 50,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Get conversation history for a user"""
    
    try:
        # Check if user can access this conversation
        if current_user and current_user.get("user_id") != user_id:
            if not current_user.get("is_admin", False):
                raise HTTPException(status_code=403, detail="Access denied")
        
        context = chat_service.get_conversation_context(user_id)
        if not context:
            return {"messages": [], "conversation_id": None}
        
        recent_messages = context.get_recent_messages(limit)
        
        return {
            "conversation_id": context.conversation_id,
            "messages": [msg.dict() for msg in recent_messages],
            "language_preference": context.language_preference,
            "created_at": context.created_at,
            "updated_at": context.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation history"
        )

@router.delete("/conversation/{user_id}")
async def clear_conversation(
    user_id: str,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Clear conversation history for a user"""
    
    try:
        # Check permissions
        if current_user and current_user.get("user_id") != user_id:
            if not current_user.get("is_admin", False):
                raise HTTPException(status_code=403, detail="Access denied")
        
        success = await chat_service.clear_conversation(user_id)
        
        return {
            "success": success,
            "message": "Conversation cleared successfully" if success else "No conversation found"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear conversation"
        )

@router.post("/voice")
@rate_limit(requests_per_minute=10)
async def process_voice_message(
    audio_file: UploadFile = File(...),
    language: Optional[str] = None,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Process voice message and return text response"""
    
    try:
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format"
            )
        
        user_id = current_user.get("user_id") if current_user else "anonymous"
        
        # Read audio file
        audio_content = await audio_file.read()
        
        # This would implement speech-to-text conversion
        # For now, return a placeholder response
        text_message = "Voice message received - speech recognition not implemented yet"
        
        response = await chat_service.process_message(
            user_id=user_id,
            message=text_message,
            language=language or "en"
        )
        
        return {
            "transcribed_text": text_message,
            "response": response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process voice message"
        )

@router.post("/translate")
@rate_limit(requests_per_minute=60)
async def translate_message(
    text: str,
    target_language: str,
    source_language: Optional[str] = None
):
    """Translate text to target language"""
    
    try:
        from app.services.translation_service import TranslationService
        translation_service = TranslationService()
        
        translated_text = await translation_service.translate(
            text=text,
            target_language=target_language,
            source_language=source_language
        )
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": source_language,
            "target_language": target_language
        }
        
    except Exception as e:
        logger.error(f"Error translating message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Translation failed"
        )

@router.get("/suggestions")
async def get_suggestions(
    language: str = "en",
    category: Optional[str] = None
):
    """Get suggested questions/prompts"""
    
    suggestions_data = {
        "en": {
            "general": [
                "What are the fee payment deadlines?",
                "How do I apply for scholarships?",
                "Show me the academic calendar",
                "Where is the library located?",
                "How do I register for courses?"
            ],
            "fees": [
                "When is the next fee deadline?",
                "How can I pay fees online?",
                "What are the late fee charges?",
                "Can I pay fees in installments?"
            ],
            "academics": [
                "What is the exam schedule?",
                "How do I check my grades?",
                "When do classes start?",
                "What are the course requirements?"
            ]
        },
        "hi": {
            "general": [
                "फीस भुगतान की अंतिम तारीख क्या है?",
                "मैं छात्रवृत्ति के लिए कैसे आवेदन करूं?",
                "मुझे शैक्षणिक कैलेंडर दिखाएं",
                "पुस्तकालय कहाँ स्थित है?",
                "मैं पाठ्यक्रमों के लिए पंजीकरण कैसे करूं?"
            ]
        }
    }
    
    lang_suggestions = suggestions_data.get(language, suggestions_data["en"])
    if category:
        return {"suggestions": lang_suggestions.get(category, lang_suggestions["general"])}
    
    return {"suggestions": lang_suggestions["general"]}

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    
    from app.config import LANGUAGE_CONFIG
    
    return {
        "supported_languages": [
            {
                "code": code,
                "name": info["name"],
                "native": info["native"]
            }
            for code, info in LANGUAGE_CONFIG.items()
        ]
    }

@router.get("/health")
async def chat_health_check():
    """Health check for chat service"""
    
    return {
        "status": "healthy",
        "service": "chat",
        "ai_models_available": bool(chat_service.openai_client or chat_service.anthropic_client),
        "translation_available": True,
        "vector_search_available": True
    }
