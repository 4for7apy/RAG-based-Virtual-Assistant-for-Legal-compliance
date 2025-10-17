"""
Language Detection Utility
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LanguageDetector:
    def __init__(self):
        # Simple keyword-based detection for demo
        self.hindi_keywords = ["क्या", "कैसे", "कहाँ", "कब", "फीस", "छात्रवृत्ति"]
        self.bengali_keywords = ["কি", "কিভাবে", "কোথায়", "কখন"]
        self.tamil_keywords = ["என்ன", "எப்படி", "எங்கே", "எப்போது"]
        self.telugu_keywords = ["ఏమిటి", "ఎలా", "ఎక్కడ", "ఎప్పుడు"]
    
    async def detect(self, text: str) -> str:
        """Detect language of input text"""
        
        text_lower = text.lower()
        
        # Check for Hindi
        if any(keyword in text for keyword in self.hindi_keywords):
            return "hi"
        
        # Check for Bengali  
        if any(keyword in text for keyword in self.bengali_keywords):
            return "bn"
            
        # Check for Tamil
        if any(keyword in text for keyword in self.tamil_keywords):
            return "ta"
            
        # Check for Telugu
        if any(keyword in text for keyword in self.telugu_keywords):
            return "te"
        
        # Default to English
        return "en"
