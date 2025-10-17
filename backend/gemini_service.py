"""
Gemini AI service for generating responses using Google's Gemini API
"""
import google.generativeai as genai
from typing import List, Dict, Optional
import os
from app.config import settings

class GeminiService:
    def __init__(self):
        self.model = None
        self.initialized = False
    
    def initialize(self):
        """Initialize Gemini service with API key"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("⚠️  GEMINI_API_KEY not found. Please set it in your .env file")
                return False
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.initialized = True
            print("✅ Gemini service initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Gemini service: {e}")
            return False
    
    async def generate_response(
        self, 
        user_message: str, 
        context: str = "", 
        conversation_history: List[Dict] = None,
        language: str = "en"
    ) -> Dict:
        """
        Generate response using Gemini
        
        Args:
            user_message: User's input message
            context: RAG context from documents
            conversation_history: Previous conversation messages
            language: Target language for response
            
        Returns:
            Dict with response, confidence, and metadata
        """
        if not self.initialized:
            return {
                "response": "Sorry, the AI service is not available right now.",
                "confidence": 0.0,
                "source": "error",
                "metadata": {"error": "Gemini not initialized"}
            }
        
        try:
            # Build the prompt
            prompt = self._build_prompt(user_message, context, conversation_history, language)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return {
                "response": response.text,
                "confidence": 0.9,  # Gemini typically has high confidence
                "source": "gemini",
                "metadata": {
                    "model": "gemini-pro",
                    "language": language,
                    "has_context": bool(context)
                }
            }
            
        except Exception as e:
            print(f"❌ Error generating Gemini response: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your request. Please try again.",
                "confidence": 0.0,
                "source": "error",
                "metadata": {"error": str(e)}
            }
    
    def _build_prompt(
        self, 
        user_message: str, 
        context: str, 
        conversation_history: List[Dict], 
        language: str
    ) -> str:
        """Build the prompt for Gemini"""
        
        # Language-specific instructions
        language_instructions = {
            "en": "Respond in English",
            "hi": "Respond in Hindi (हिंदी)",
            "ta": "Respond in Tamil (தமிழ்)",
            "te": "Respond in Telugu (తెలుగు)",
            "mr": "Respond in Marathi (मराठी)"
        }
        
        lang_instruction = language_instructions.get(language, "Respond in English")
        
        prompt = f"""You are a helpful campus assistant chatbot for a university. {lang_instruction}

Your role:
- Help students with campus-related queries
- Provide accurate information about admissions, fees, academics, etc.
- Be friendly, professional, and helpful
- If you don't know something, say so clearly

"""
        
        # Add context if available
        if context:
            prompt += f"""
Relevant information from campus documents:
{context}

"""
        
        # Add conversation history if available
        if conversation_history:
            prompt += "Previous conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = "Student" if msg.get("role") == "user" else "Assistant"
                prompt += f"{role}: {msg.get('content', '')}\n"
            prompt += "\n"
        
        # Add current user message
        prompt += f"Current student question: {user_message}\n\n"
        prompt += "Please provide a helpful response:"
        
        return prompt
