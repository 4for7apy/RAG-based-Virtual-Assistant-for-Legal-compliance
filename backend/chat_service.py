"""
Advanced Chat Service with Multilingual Support
Handles AI-powered conversations with context management
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import openai
from anthropic import Anthropic
from googletrans import Translator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.config import settings, LANGUAGE_CONFIG
from app.models.conversation import ConversationContext, Message
from app.services.gemini_service import GeminiService
from app.services.translation_service import TranslationService
from app.services.vector_service import VectorService
from app.services.rag_service import rag_service
from app.services.advanced_embeddings import embeddings_service, vector_store
from app.utils.language_detector import LanguageDetector
from app.utils.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)

class ChatService:
    """Advanced chat service with multilingual AI support"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_service = GeminiService()
        self.translation_service = TranslationService()
        self.vector_service = VectorService()
        self.language_detector = LanguageDetector()
        self.intent_classifier = IntentClassifier()
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
    async def initialize(self):
        """Initialize AI clients and services"""
        try:
            if settings.OPENAI_API_KEY:
                openai.api_key = settings.OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized")
            
            if settings.ANTHROPIC_API_KEY:
                self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            
            # Initialize RAG and embedding services
            await rag_service.initialize()
            await embeddings_service.initialize()
            
            # Initialize Gemini service
            self.gemini_service.initialize()
            
            await self.vector_service.initialize()
            await self.translation_service.initialize()
            
            logger.info("Chat service with RAG initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {str(e)}")
            raise
    
    async def process_message(
        self, 
        user_id: str, 
        message: str, 
        language: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process incoming message and generate AI response"""
        
        try:
            # Get or create conversation context
            if user_id not in self.conversation_contexts:
                self.conversation_contexts[user_id] = ConversationContext(user_id)
            
            conv_context = self.conversation_contexts[user_id]
            
            # Detect language if not provided
            if not language:
                language = await self.language_detector.detect(message)
            
            # Store user message
            user_message = Message(
                content=message,
                sender="user",
                language=language,
                timestamp=datetime.now()
            )
            conv_context.add_message(user_message)
            
            # Classify intent
            intent = await self.intent_classifier.classify(message, language)
            
            # Use RAG to get intelligent context
            rag_response = await rag_service.generate_rag_response(
                query=message,
                language=language
            )
            
            # If RAG has a good response, use it directly
            if rag_response["confidence"] > 0.8:
                ai_message = Message(
                    content=rag_response["response"],
                    sender="assistant",
                    language=language,
                    metadata={
                        "intent": intent,
                        "confidence": rag_response["confidence"],
                        "source": "rag",
                        "model_used": "rag_system",
                        "relevant_documents": rag_response["relevant_documents"]
                    },
                    timestamp=datetime.now()
                )
                conv_context.add_message(ai_message)
                
                # Log conversation for analytics
                await self._log_conversation(user_id, user_message, ai_message, intent)
                
                return {
                    "response": rag_response["response"],
                    "language": language,
                    "intent": intent,
                    "confidence": rag_response["confidence"],
                    "suggestions": await self._generate_suggestions(intent, language),
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": conv_context.conversation_id,
                    "source": "rag",
                    "relevant_documents": rag_response["relevant_documents"]
                }
            
            # Fallback to traditional knowledge search
            knowledge_context = rag_response["relevant_documents"]
            
            # Generate AI response
            ai_response = await self.generate_response(
                message=message,
                language=language,
                intent=intent,
                conversation_history=conv_context.get_recent_messages(10),
                knowledge_context=knowledge_context,
                user_context=context or {}
            )
            
            # Store AI response
            ai_message = Message(
                content=ai_response["content"],
                sender="assistant",
                language=language,
                metadata={
                    "intent": intent,
                    "confidence": ai_response.get("confidence", 0.8),
                    "model_used": ai_response.get("model_used", "unknown")
                },
                timestamp=datetime.now()
            )
            conv_context.add_message(ai_message)
            
            # Log conversation for analytics
            await self._log_conversation(user_id, user_message, ai_message, intent)
            
            return {
                "response": ai_response["content"],
                "language": language,
                "intent": intent,
                "confidence": ai_response.get("confidence", 0.8),
                "suggestions": await self._generate_suggestions(intent, language),
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conv_context.conversation_id
            }
            
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {str(e)}")
            return await self._generate_error_response(language or "en")
    
    async def generate_response(
        self,
        message: str,
        language: str,
        intent: str,
        conversation_history: List[Message],
        knowledge_context: List[Dict],
        user_context: Dict
    ) -> Dict[str, Any]:
        """Generate AI response using the best available model"""
        
        # Build context for AI model
        system_prompt = await self._build_system_prompt(language, intent)
        context_prompt = await self._build_context_prompt(
            knowledge_context, conversation_history
        )
        
        # Choose the best model based on query complexity
        model = await self._select_model(message, intent)
        
        try:
            if model.startswith("gpt"):
                response = await self._generate_openai_response(
                    system_prompt, context_prompt, message, model, language
                )
            elif model.startswith("claude"):
                response = await self._generate_anthropic_response(
                    system_prompt, context_prompt, message, model, language
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
            
            # Translate response if needed
            if language != "en":
                translated_response = await self.translation_service.translate(
                    response["content"], target_language=language
                )
                response["content"] = translated_response
            
            response["model_used"] = model
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with {model}: {str(e)}")
            # Fallback to simpler model
            return await self._generate_fallback_response(message, language)
    
    async def _generate_openai_response(
        self, system_prompt: str, context_prompt: str, message: str, 
        model: str, language: str
    ) -> Dict[str, Any]:
        """Generate response using OpenAI models"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": message}
        ]
        
        response = await self.openai_client.ChatCompletion.acreate(
            model=model,
            messages=messages,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        return {
            "content": response.choices[0].message.content,
            "confidence": 0.9,
            "tokens_used": response.usage.total_tokens
        }
    
    async def _generate_anthropic_response(
        self, system_prompt: str, context_prompt: str, message: str, 
        model: str, language: str
    ) -> Dict[str, Any]:
        """Generate response using Anthropic Claude"""
        
        prompt = f"{system_prompt}\n\n{context_prompt}\n\nHuman: {message}\n\nAssistant:"
        
        response = await self.anthropic_client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens_to_sample=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE
        )
        
        return {
            "content": response.completion.strip(),
            "confidence": 0.85,
            "tokens_used": len(response.completion.split())
        }
    
    async def _build_system_prompt(self, language: str, intent: str) -> str:
        """Build system prompt based on language and intent"""
        
        lang_info = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["en"])
        
        base_prompt = f"""You are a helpful campus assistant chatbot for students and faculty.
        
        LANGUAGE: Respond in {lang_info['native']} ({lang_info['name']})
        INTENT: The user's query is classified as '{intent}'
        
        GUIDELINES:
        - Be helpful, concise, and friendly
        - Provide accurate information about campus services
        - If you're unsure, ask for clarification or suggest contacting human support
        - Use the same language as the user's query
        - For complex queries, break down the response into clear steps
        - Always maintain a professional yet approachable tone
        
        AVAILABLE SERVICES:
        - Fee information and payment deadlines
        - Scholarship applications and eligibility
        - Academic calendar and timetables
        - Campus facilities and locations
        - Admission procedures
        - Examination schedules
        - Hostel and accommodation
        - Library services
        - Student activities and clubs
        """
        
        return base_prompt
    
    async def _build_context_prompt(
        self, knowledge_context: List[Dict], conversation_history: List[Message]
    ) -> str:
        """Build context prompt from knowledge base and conversation history"""
        
        context_parts = []
        
        # Add relevant knowledge base information
        if knowledge_context:
            context_parts.append("RELEVANT INFORMATION:")
            for i, ctx in enumerate(knowledge_context[:3], 1):
                context_parts.append(f"{i}. {ctx.get('content', '')[:500]}...")
        
        # Add recent conversation history
        if conversation_history:
            context_parts.append("\nRECENT CONVERSATION:")
            for msg in conversation_history[-5:]:
                role = "Student" if msg.sender == "user" else "Assistant"
                context_parts.append(f"{role}: {msg.content[:200]}...")
        
        return "\n".join(context_parts)
    
    async def _select_model(self, message: str, intent: str) -> str:
        """Select the best AI model based on query complexity"""
        
        # Simple heuristics for model selection
        complex_intents = ["academic_advice", "complex_query", "multi_step"]
        long_query = len(message.split()) > 20
        
        if intent in complex_intents or long_query:
            return settings.PRIMARY_MODEL
        else:
            return settings.FALLBACK_MODEL
    
    async def _generate_suggestions(self, intent: str, language: str) -> List[str]:
        """Generate follow-up suggestions based on intent"""
        
        suggestions_map = {
            "fee_inquiry": [
                "When is the next fee deadline?",
                "How can I pay fees online?",
                "What are the late fee charges?"
            ],
            "scholarship": [
                "What documents are needed?",
                "When is the application deadline?",
                "Am I eligible for other scholarships?"
            ],
            "timetable": [
                "Show exam schedule",
                "When are holidays?",
                "What time does library close?"
            ]
        }
        
        suggestions = suggestions_map.get(intent, [
            "Tell me about campus facilities",
            "How do I contact support?",
            "What other services are available?"
        ])
        
        # Translate suggestions if needed
        if language != "en":
            translated_suggestions = []
            for suggestion in suggestions:
                translated = await self.translation_service.translate(
                    suggestion, target_language=language
                )
                translated_suggestions.append(translated)
            return translated_suggestions
        
        return suggestions
    
    async def _generate_error_response(self, language: str) -> Dict[str, Any]:
        """Generate error response in appropriate language"""
        
        error_messages = {
            "en": "I'm sorry, I encountered an error. Please try again or contact support.",
            "hi": "माफ़ करें, मुझे एक त्रुटि का सामना करना पड़ा। कृपया पुनः प्रयास करें या सहायता से संपर्क करें।",
            "bn": "দুঃখিত, আমি একটি ত্রুটির সম্মুখীন হয়েছি। অনুগ্রহ করে আবার চেষ্টা করুন বা সহায়তার সাথে যোগাযোগ করুন।",
            "ta": "மன்னிக்கவும், நான் ஒரு பிழையை எதிர்கொண்டேன். தயவுசெய்து மீண்டும் முயற்சிக்கவும் அல்லது ஆதரவைத் தொடர்பு கொள்ளவும்.",
            "te": "క్షమించండి, నేను లోపాన్ని ఎదుర్కొన్నాను. దయచేసి మళ్లీ ప్రయత్నించండి లేదా మద్దతును సంప్రదించండి."
        }
        
        return {
            "response": error_messages.get(language, error_messages["en"]),
            "language": language,
            "intent": "error",
            "confidence": 1.0,
            "suggestions": [],
            "timestamp": datetime.now().isoformat(),
            "error": True
        }
    
    async def _generate_fallback_response(self, message: str, language: str) -> Dict[str, Any]:
        """Generate simple fallback response"""
        
        fallback_messages = {
            "en": "I understand you're asking about campus services. Could you please be more specific so I can help you better?",
            "hi": "मैं समझ गया हूं कि आप कैंपस सेवाओं के बारे में पूछ रहे हैं। कृपया अधिक विशिष्ट बताएं ताकि मैं आपकी बेहतर सहायता कर सकूं?",
            "bn": "আমি বুঝতে পারছি আপনি ক্যাম্পাস সেবা সম্পর্কে জিজ্ঞাসা করছেন। আমি আপনাকে আরও ভালভাবে সাহায্য করতে পারি এমন আরও নির্দিষ্ট বিষয় বলুন।",
            "ta": "நீங்கள் வளாக சேவைகளைப் பற்றி கேட்கிறீர்கள் என்று நான் புரிந்துகொள்கிறேன். நான் உங்களுக்கு சிறப்பாக உதவ முடியும் என்று அधிக குறிப்பிட்ட தகவல் கொடுக்க முடியுமா?",
            "te": "మీరు క్యాంపస్ సేవల గురించి అడుగుతున్నారని నేను అర్థం చేసుకున్నాను. నేను మీకు మరింత మెరుగ్గా సహాయం చేయగలిగేలా దయచేసి మరింత నిర్దిష్టంగా చెప్పగలరా?"
        }
        
        return {
            "content": fallback_messages.get(language, fallback_messages["en"]),
            "confidence": 0.6
        }
    
    async def _log_conversation(
        self, user_id: str, user_message: Message, 
        ai_message: Message, intent: str
    ):
        """Log conversation for analytics and improvement"""
        
        log_data = {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "user_message": {
                "content": user_message.content,
                "language": user_message.language
            },
            "ai_response": {
                "content": ai_message.content,
                "language": ai_message.language,
                "metadata": ai_message.metadata
            },
            "intent": intent,
            "session_id": self.conversation_contexts[user_id].conversation_id
        }
        
        # This would typically save to MongoDB or another logging service
        logger.info(f"Conversation logged for user {user_id}")
    
    def get_conversation_context(self, user_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a user"""
        return self.conversation_contexts.get(user_id)
    
    async def clear_conversation(self, user_id: str) -> bool:
        """Clear conversation history for a user"""
        if user_id in self.conversation_contexts:
            del self.conversation_contexts[user_id]
            return True
        return False
