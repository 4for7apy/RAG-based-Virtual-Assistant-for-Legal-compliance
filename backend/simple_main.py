#!/usr/bin/env python3
"""
Simplified FastAPI backend for Campus Chatbot with Gemini integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Campus Chatbot API",
    description="Multilingual AI-powered campus assistant",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
gemini_model = None

def initialize_gemini():
    global gemini_model
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️  GEMINI_API_KEY not found in .env file")
            return False
        
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("✅ Gemini initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        return False

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    language: str = "en"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    language: str
    confidence: float
    source: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    gemini_available: bool
    message: str

# Routes
@app.get("/", response_model=Dict)
async def root():
    return {
        "message": "Campus Chatbot API is running!",
        "status": "healthy",
        "gemini_available": gemini_model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        gemini_available=gemini_model is not None,
        message="Campus Chatbot API is running" if gemini_model else "Gemini not initialized"
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message_data: ChatMessage):
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini AI service not available")
    
    try:
        # Build prompt based on language
        language_instructions = {
            "en": "Respond in English",
            "hi": "Respond in Hindi (हिंदी)",
            "ta": "Respond in Tamil (தமிழ்)",
            "te": "Respond in Telugu (తెలుగు)",
            "mr": "Respond in Marathi (मराठी)"
        }
        
        lang_instruction = language_instructions.get(message_data.language, "Respond in English")
        
        prompt = f"""You are a helpful campus assistant chatbot for a university. {lang_instruction}

Your role:
- Help students with campus-related queries
- Provide accurate information about admissions, fees, academics, etc.
- Be friendly, professional, and helpful
- If you don't know something, say so clearly

Student question: {message_data.message}

Please provide a helpful response:"""

        # Generate response
        response = gemini_model.generate_content(prompt)
        
        return ChatResponse(
            response=response.text,
            language=message_data.language,
            confidence=0.9,
            source="gemini",
            timestamp=os.popen('date').read().strip()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/api/supported-languages")
async def get_supported_languages():
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"},
            {"code": "ta", "name": "Tamil"},
            {"code": "te", "name": "Telugu"},
            {"code": "mr", "name": "Marathi"}
        ]
    }

@app.get("/api/features")
async def get_features():
    return {
        "features": [
            "Multilingual support (5+ languages)",
            "Gemini AI-powered responses",
            "Campus-specific knowledge",
            "Real-time chat",
            "RAG document processing (coming soon)"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Campus Chatbot API...")
    
    # Initialize Gemini
    if initialize_gemini():
        print("✅ Backend ready with Gemini AI!")
    else:
        print("⚠️  Backend running without Gemini AI")
    
    import uvicorn
    uvicorn.run("simple_main:app", host="0.0.0.0", port=8000, reload=True)
