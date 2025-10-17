"""
Intent Classification Utility
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self):
        # Simple keyword-based classification for demo
        self.intent_keywords = {
            "fee_inquiry": ["fee", "payment", "tuition", "cost", "price", "फीस", "भुगतान"],
            "scholarship": ["scholarship", "financial aid", "grant", "छात्रवृत्ति"],
            "timetable": ["schedule", "timetable", "class", "timing", "समय सारणी"],
            "location": ["where", "location", "find", "कहाँ", "स्थान"],
            "admission": ["admission", "apply", "registration", "प्रवेश"],
            "academic_info": ["academic", "course", "subject", "exam", "grade"],
            "facilities": ["library", "hostel", "canteen", "gym", "facilities"],
            "contact": ["contact", "phone", "email", "address", "संपर्क"]
        }
    
    async def classify(self, text: str, language: str = "en") -> str:
        """Classify intent of input text"""
        
        text_lower = text.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            return best_intent
        
        # Default intent
        return "general_inquiry"
