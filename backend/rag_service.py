"""
RAG (Retrieval-Augmented Generation) Service
Complete implementation for intelligent document-based responses
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json

# Vector and embedding libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Text processing
import re
from pathlib import Path

# Database
from app.database import get_mongodb, get_redis

logger = logging.getLogger(__name__)

class RAGService:
    """Advanced RAG service for campus knowledge retrieval"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.document_vectors = None
        self.documents = []
        self.metadata = []
        self.is_trained = False
        self.knowledge_base_path = "data/knowledge_base"
        self.cache_path = "data/cache"
        
        # Create directories
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)
    
    async def initialize(self):
        """Initialize RAG service"""
        try:
            logger.info("Initializing RAG service...")
            
            # Load existing knowledge base
            await self.load_knowledge_base()
            
            # Create campus knowledge if not exists
            if not self.documents:
                await self.create_campus_knowledge_base()
            
            # Train the model
            await self.train_model()
            
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise
    
    async def create_campus_knowledge_base(self):
        """Create comprehensive campus knowledge base"""
        
        logger.info("Creating campus knowledge base...")
        
        campus_knowledge = {
            # Academic Information
            "academic_calendar": {
                "content": """
                Academic Calendar 2024-25:
                
                Spring Semester: January 15 - May 15, 2024
                - Registration: January 1-10, 2024
                - Classes Begin: January 15, 2024
                - Mid-term Exams: March 1-8, 2024
                - Spring Break: March 15-22, 2024
                - Final Exams: May 1-15, 2024
                
                Summer Session: June 1 - July 31, 2024
                - Registration: May 20-25, 2024
                - Classes: June 1 - July 25, 2024
                - Final Exams: July 26-31, 2024
                
                Fall Semester: August 15 - December 15, 2024
                - Registration: August 1-10, 2024
                - Classes Begin: August 15, 2024
                - Mid-term Exams: October 1-8, 2024
                - Thanksgiving Break: November 25-29, 2024
                - Final Exams: December 1-15, 2024
                
                Winter Break: December 16, 2024 - January 14, 2025
                """,
                "category": "academics",
                "keywords": ["calendar", "semester", "dates", "registration", "exams"]
            },
            
            # Fee Information
            "fee_structure": {
                "content": """
                Fee Structure 2024-25:
                
                Tuition Fees (per semester):
                - Undergraduate: ₹75,000
                - Postgraduate: ₹90,000
                - PhD: ₹50,000
                
                Additional Fees:
                - Registration Fee: ₹5,000 (one-time)
                - Library Fee: ₹2,000 per semester
                - Lab Fee: ₹3,000 per semester (for science/engineering)
                - Sports Fee: ₹1,500 per semester
                - Development Fee: ₹10,000 per year
                
                Hostel Fees (per semester):
                - Single Room: ₹25,000
                - Double Room: ₹18,000
                - Triple Room: ₹15,000
                - Mess Charges: ₹8,000 per semester
                
                Payment Deadlines:
                - Tuition Fees: January 15th and August 15th
                - Hostel Fees: January 10th and August 10th
                - Late Fee: ₹500 per day after deadline
                
                Payment Methods:
                - Online through student portal
                - Bank transfer
                - Demand Draft
                - Cash at accounts office
                """,
                "category": "fees",
                "keywords": ["fee", "tuition", "payment", "deadline", "cost", "hostel"]
            },
            
            # Scholarship Information
            "scholarships": {
                "content": """
                Scholarship Programs 2024-25:
                
                Merit-Based Scholarships:
                - Excellence Scholarship: 100% tuition waiver (GPA > 9.5)
                - Merit Scholarship: 50% tuition waiver (GPA > 8.5)
                - Academic Achievement: 25% tuition waiver (GPA > 7.5)
                
                Need-Based Scholarships:
                - Economic Support: Up to ₹50,000 per year
                - Family Income Criteria: < ₹3,00,000 per annum
                - Documentation Required: Income certificate, bank statements
                
                Special Category Scholarships:
                - SC/ST Scholarship: 100% fee waiver + ₹20,000 stipend
                - OBC Scholarship: 50% fee waiver
                - Minority Scholarship: 30% fee waiver
                - Sports Scholarship: Up to ₹75,000 for state/national players
                - Cultural Scholarship: Up to ₹25,000 for exceptional talent
                
                Application Process:
                - Online application through student portal
                - Required documents: Mark sheets, income proof, certificates
                - Application deadline: December 30th for next academic year
                - Selection process: Document verification + interview
                - Results announced: February 15th
                
                Renewal Criteria:
                - Maintain minimum GPA of 7.0
                - Regular attendance (>75%)
                - No disciplinary issues
                """,
                "category": "scholarships",
                "keywords": ["scholarship", "financial aid", "merit", "need-based", "application"]
            },
            
            # Campus Facilities
            "facilities": {
                "content": """
                Campus Facilities:
                
                Library Services:
                - Central Library: 24/7 access with AC study halls
                - Digital Library: 50,000+ e-books and journals
                - Research Section: Dedicated PhD research area
                - Group Study Rooms: Bookable online
                - Printing & Scanning: Available 8 AM - 10 PM
                - Library Hours: 6 AM - 12 AM (Monday-Friday), 8 AM - 10 PM (Weekends)
                
                Computer Labs:
                - 5 Computer Labs with 200+ systems
                - Software: MATLAB, AutoCAD, Visual Studio, Adobe Suite
                - Internet Speed: 1 Gbps fiber connection
                - Printing Facility: Laser printers available
                - Access Hours: 8 AM - 10 PM
                
                Sports Complex:
                - Football Ground: FIFA standard artificial turf
                - Basketball Courts: 2 indoor courts
                - Tennis Courts: 4 outdoor courts
                - Swimming Pool: Olympic size, heated
                - Gymnasium: Modern equipment, personal trainers
                - Indoor Games: Table tennis, badminton, chess
                
                Health Services:
                - Medical Center: 24/7 emergency services
                - Qualified Doctors: Available 9 AM - 6 PM
                - Pharmacy: Basic medicines available
                - Ambulance Service: Emergency transportation
                - Health Insurance: Covered for all students
                
                Dining Facilities:
                - Main Cafeteria: 500 seating capacity
                - Food Courts: 10 different cuisine options
                - Coffee Shops: 3 locations across campus
                - Vending Machines: Snacks and beverages
                - Catering Services: For events and functions
                
                Other Facilities:
                - ATM: 3 ATMs from different banks
                - Post Office: Full postal services
                - Stationery Shop: Academic supplies
                - Laundry Service: Hostel laundry facility
                - Transportation: Campus shuttle service
                - Parking: Separate areas for cars and bikes
                """,
                "category": "facilities",
                "keywords": ["library", "computer lab", "sports", "medical", "cafeteria", "ATM"]
            },
            
            # Admission Information
            "admissions": {
                "content": """
                Admission Process 2024-25:
                
                Undergraduate Programs:
                - Application Period: March 1 - May 31, 2024
                - Entrance Exam: June 15, 2024
                - Merit List: July 1, 2024
                - Counseling: July 5-20, 2024
                - Document Verification: July 25-30, 2024
                - Fee Payment Deadline: August 5, 2024
                
                Eligibility Criteria:
                - 12th Standard with 75% marks (70% for reserved categories)
                - Science stream for Engineering courses
                - Any stream for Management courses
                - English proficiency required
                
                Required Documents:
                - 10th and 12th mark sheets
                - Character certificate
                - Migration certificate
                - Caste certificate (if applicable)
                - Income certificate (for scholarships)
                - Medical fitness certificate
                - Passport size photographs
                
                Postgraduate Programs:
                - Application Period: April 1 - June 30, 2024
                - Entrance Exam: July 15, 2024
                - Interview: July 25-30, 2024
                - Final Merit List: August 5, 2024
                
                International Students:
                - Separate application process
                - English proficiency test required
                - Visa assistance provided
                - Dedicated international student office
                """,
                "category": "admissions",
                "keywords": ["admission", "application", "entrance", "eligibility", "documents"]
            },
            
            # Hostel Information
            "hostel_info": {
                "content": """
                Hostel Accommodation:
                
                Boys Hostels:
                - Hostel A: 200 rooms, single occupancy
                - Hostel B: 150 rooms, double occupancy
                - Hostel C: 100 rooms, triple occupancy
                
                Girls Hostels:
                - Hostel D: 180 rooms, single occupancy
                - Hostel E: 120 rooms, double occupancy
                - Hostel F: 80 rooms, triple occupancy
                
                Room Facilities:
                - Bed, study table, chair, wardrobe
                - Wi-Fi internet connection
                - 24/7 electricity and water supply
                - Common areas with TV and recreation
                
                Hostel Rules:
                - Entry timings: Boys - 11 PM, Girls - 9 PM
                - Visitors allowed in common areas only
                - No outside food in rooms
                - Smoking and alcohol strictly prohibited
                - Room inspection every month
                
                Mess Facilities:
                - Vegetarian and non-vegetarian options
                - 4 meals per day included
                - Special dietary requirements accommodated
                - Monthly menu available online
                - Feedback system for food quality
                
                Application Process:
                - Online application through student portal
                - Room allotment based on merit and distance
                - Advance payment required for confirmation
                - Check-in during orientation week
                """,
                "category": "hostel",
                "keywords": ["hostel", "accommodation", "rooms", "mess", "facilities"]
            },
            
            # Academic Programs
            "programs": {
                "content": """
                Academic Programs:
                
                Undergraduate Programs (4 years):
                - B.Tech Computer Science Engineering
                - B.Tech Electronics & Communication
                - B.Tech Mechanical Engineering
                - B.Tech Civil Engineering
                - B.Sc Physics, Chemistry, Mathematics
                - B.Com Commerce and Finance
                - BBA Business Administration
                - BA English Literature
                
                Postgraduate Programs (2 years):
                - M.Tech Computer Science
                - M.Tech Electronics & Communication
                - MBA (Marketing, Finance, HR, Operations)
                - M.Sc Physics, Chemistry, Mathematics
                - M.Com Advanced Commerce
                - MA English Literature
                
                Doctoral Programs (3-5 years):
                - PhD in Engineering disciplines
                - PhD in Sciences
                - PhD in Management
                - PhD in Humanities
                
                Certificate Courses:
                - Data Science and Analytics
                - Digital Marketing
                - Foreign Languages (German, French, Spanish)
                - Soft Skills Development
                - Entrepreneurship Development
                
                Industry Partnerships:
                - Internship programs with top companies
                - Guest lectures by industry experts
                - Live project opportunities
                - Placement assistance
                - Industry-academia collaboration
                """,
                "category": "academics",
                "keywords": ["programs", "courses", "degrees", "engineering", "management"]
            }
        }
        
        # Save knowledge to files and add to documents
        for knowledge_id, knowledge_data in campus_knowledge.items():
            # Save to file
            file_path = os.path.join(self.knowledge_base_path, f"{knowledge_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
            
            # Add to documents
            self.documents.append(knowledge_data["content"])
            self.metadata.append({
                "id": knowledge_id,
                "category": knowledge_data["category"],
                "keywords": knowledge_data["keywords"],
                "source": "campus_knowledge_base",
                "created_at": datetime.now().isoformat()
            })
        
        logger.info(f"Created {len(campus_knowledge)} knowledge base documents")
    
    async def load_knowledge_base(self):
        """Load existing knowledge base from files"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                return
            
            for file_path in Path(self.knowledge_base_path).glob("*.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                
                self.documents.append(knowledge_data["content"])
                metadata = {
                    "id": file_path.stem,
                    "category": knowledge_data.get("category", "general"),
                    "keywords": knowledge_data.get("keywords", []),
                    "source": "file",
                    "file_path": str(file_path)
                }
                self.metadata.append(metadata)
            
            logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
    
    async def train_model(self):
        """Train the vector model on documents"""
        try:
            if not self.documents:
                logger.warning("No documents available for training")
                return
            
            logger.info("Training RAG model...")
            
            # Preprocess documents
            processed_docs = [self.preprocess_text(doc) for doc in self.documents]
            
            # Train TF-IDF vectorizer
            self.document_vectors = self.vectorizer.fit_transform(processed_docs)
            
            # Save the trained model
            model_path = os.path.join(self.cache_path, "rag_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'document_vectors': self.document_vectors,
                    'metadata': self.metadata
                }, f)
            
            self.is_trained = True
            logger.info(f"RAG model trained with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error training RAG model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\-\:\;]', '', text)
        
        return text.strip()
    
    async def retrieve_relevant_documents(
        self, 
        query: str, 
        top_k: int = 3,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for a query"""
        
        try:
            if not self.is_trained or self.document_vectors is None:
                logger.warning("RAG model not trained")
                return []
            
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            # Transform query to vector
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top-k most similar documents
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            relevant_docs = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > similarity_threshold:
                    relevant_docs.append({
                        "content": self.documents[idx],
                        "metadata": self.metadata[idx],
                        "similarity": float(similarity),
                        "relevance_score": float(similarity * 100)
                    })
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query: {query[:50]}...")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def generate_rag_response(
        self, 
        query: str, 
        language: str = "en",
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """Generate RAG-enhanced response"""
        
        try:
            # Retrieve relevant documents
            relevant_docs = await self.retrieve_relevant_documents(query, top_k=3)
            
            if not relevant_docs:
                return {
                    "response": "I don't have specific information about that. Please contact the administration for more details.",
                    "source": "fallback",
                    "confidence": 0.3,
                    "relevant_documents": []
                }
            
            # Combine context from relevant documents
            context_parts = []
            sources = []
            
            for doc in relevant_docs:
                # Truncate content if too long
                content = doc["content"]
                if len(content) > 500:
                    content = content[:500] + "..."
                
                context_parts.append(f"[Source: {doc['metadata']['category']}]\n{content}")
                sources.append({
                    "category": doc["metadata"]["category"],
                    "relevance": doc["relevance_score"],
                    "keywords": doc["metadata"].get("keywords", [])
                })
            
            # Create combined context
            combined_context = "\n\n".join(context_parts)
            if len(combined_context) > max_context_length:
                combined_context = combined_context[:max_context_length] + "..."
            
            # Generate response based on context
            response = await self.generate_contextual_response(
                query, combined_context, language, relevant_docs
            )
            
            return {
                "response": response,
                "source": "rag",
                "confidence": 0.9,
                "relevant_documents": sources,
                "context_used": len(combined_context)
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return {
                "response": "I encountered an error processing your question. Please try again.",
                "source": "error",
                "confidence": 0.1,
                "relevant_documents": []
            }
    
    async def generate_contextual_response(
        self, 
        query: str, 
        context: str, 
        language: str,
        relevant_docs: List[Dict]
    ) -> str:
        """Generate contextual response from retrieved documents"""
        
        # Simple rule-based response generation
        # In a real implementation, this would use GPT/Claude with the context
        
        query_lower = query.lower()
        
        # Fee-related queries
        if any(word in query_lower for word in ["fee", "cost", "payment", "tuition", "फीस", "भुगतान"]):
            fee_info = self.extract_fee_information(context)
            if language == "hi":
                return f"शुल्क की जानकारी:\n{fee_info}\n\nअधिक जानकारी के लिए खाता कार्यालय से संपर्क करें।"
            return f"Here's the fee information:\n{fee_info}\n\nFor more details, contact the accounts office."
        
        # Scholarship queries
        elif any(word in query_lower for word in ["scholarship", "financial aid", "छात्रवृत्ति"]):
            scholarship_info = self.extract_scholarship_information(context)
            if language == "hi":
                return f"छात्रवृत्ति की जानकारी:\n{scholarship_info}\n\nआवेदन के लिए छात्र पोर्टल पर जाएं।"
            return f"Scholarship information:\n{scholarship_info}\n\nApply through the student portal."
        
        # Facility queries
        elif any(word in query_lower for word in ["library", "lab", "hostel", "facility", "पुस्तकालय"]):
            facility_info = self.extract_facility_information(context, query_lower)
            if language == "hi":
                return f"सुविधा की जानकारी:\n{facility_info}"
            return f"Facility information:\n{facility_info}"
        
        # Academic queries
        elif any(word in query_lower for word in ["admission", "course", "program", "semester", "प्रवेश"]):
            academic_info = self.extract_academic_information(context, query_lower)
            if language == "hi":
                return f"शैक्षणिक जानकारी:\n{academic_info}"
            return f"Academic information:\n{academic_info}"
        
        # General response
        else:
            summary = self.extract_relevant_summary(context, query_lower)
            if language == "hi":
                return f"आपके प्रश्न के अनुसार:\n{summary}\n\nअधिक सहायता के लिए प्रशासन से संपर्क करें।"
            return f"Based on your question:\n{summary}\n\nFor further assistance, contact the administration."
    
    def extract_fee_information(self, context: str) -> str:
        """Extract fee-related information from context"""
        lines = context.split('\n')
        fee_lines = [line.strip() for line in lines if any(word in line.lower() for word in ['fee', 'tuition', 'cost', '₹', 'payment', 'deadline'])]
        return '\n'.join(fee_lines[:10])  # Limit to 10 most relevant lines
    
    def extract_scholarship_information(self, context: str) -> str:
        """Extract scholarship-related information from context"""
        lines = context.split('\n')
        scholarship_lines = [line.strip() for line in lines if any(word in line.lower() for word in ['scholarship', 'merit', 'financial', 'aid', 'application', 'gpa'])]
        return '\n'.join(scholarship_lines[:10])
    
    def extract_facility_information(self, context: str, query: str) -> str:
        """Extract facility-related information from context"""
        lines = context.split('\n')
        
        # Specific facility lookup
        if 'library' in query:
            facility_lines = [line.strip() for line in lines if 'library' in line.lower()]
        elif 'hostel' in query:
            facility_lines = [line.strip() for line in lines if 'hostel' in line.lower()]
        elif 'lab' in query:
            facility_lines = [line.strip() for line in lines if 'lab' in line.lower()]
        else:
            facility_lines = [line.strip() for line in lines if any(word in line.lower() for word in ['library', 'lab', 'sports', 'medical', 'cafeteria'])]
        
        return '\n'.join(facility_lines[:10])
    
    def extract_academic_information(self, context: str, query: str) -> str:
        """Extract academic-related information from context"""
        lines = context.split('\n')
        academic_lines = [line.strip() for line in lines if any(word in line.lower() for word in ['admission', 'course', 'program', 'semester', 'exam', 'calendar'])]
        return '\n'.join(academic_lines[:10])
    
    def extract_relevant_summary(self, context: str, query: str) -> str:
        """Extract most relevant information as summary"""
        lines = context.split('\n')
        relevant_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
        return '\n'.join(relevant_lines[:8])  # Return top 8 relevant lines
    
    async def add_document(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a new document to the knowledge base"""
        try:
            # Add to documents
            self.documents.append(content)
            self.metadata.append(metadata)
            
            # Save to file if has id
            if 'id' in metadata:
                file_path = os.path.join(self.knowledge_base_path, f"{metadata['id']}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "content": content,
                        "category": metadata.get("category", "general"),
                        "keywords": metadata.get("keywords", [])
                    }, f, indent=2, ensure_ascii=False)
            
            # Retrain model
            await self.train_model()
            
            logger.info(f"Added new document: {metadata.get('id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False
    
    async def search_documents(
        self, 
        query: str, 
        category: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search documents with optional category filter"""
        
        documents = await self.retrieve_relevant_documents(query, top_k * 2)
        
        if category:
            documents = [doc for doc in documents if doc['metadata'].get('category') == category]
        
        return documents[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        categories = {}
        for meta in self.metadata:
            cat = meta.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "categories": categories,
            "is_trained": self.is_trained,
            "vector_dimensions": self.document_vectors.shape[1] if self.document_vectors is not None else 0
        }

# Global RAG service instance
rag_service = RAGService()
