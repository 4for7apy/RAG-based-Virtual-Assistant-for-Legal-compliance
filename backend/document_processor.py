"""
Document Processing Pipeline for RAG System
Handles PDF, DOC, TXT files and extracts knowledge
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import hashlib
import mimetypes
import tempfile

# Document processing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import pandas as pd
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False

# Text processing
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.rag_service import rag_service
from app.services.advanced_embeddings import embeddings_service, vector_store
from app.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Advanced document processing for RAG system"""
    
    def __init__(self):
        self.supported_formats = {
            'text/plain': 'txt',
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'application/vnd.ms-excel': 'xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx'
        }
        
        self.chunk_size = 1000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
        # Create processing directories
        self.upload_dir = "data/uploads"
        self.processed_dir = "data/processed"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    async def process_file(
        self, 
        file_path: str, 
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process uploaded file and extract knowledge"""
        
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Extract text from file
            extracted_text = await self.extract_text(file_path, file_type)
            
            if not extracted_text.strip():
                return {
                    "success": False,
                    "error": "No text could be extracted from the file"
                }
            
            # Split into chunks
            chunks = self.split_text_into_chunks(extracted_text)
            
            # Process metadata
            file_metadata = metadata or {}
            file_metadata.update({
                "file_path": file_path,
                "file_type": file_type,
                "processed_at": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "total_characters": len(extracted_text)
            })
            
            # Add chunks to RAG system
            added_documents = 0
            for i, chunk in enumerate(chunks):
                chunk_metadata = file_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{Path(file_path).stem}_chunk_{i}",
                    "chunk_index": i,
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk
                })
                
                success = await rag_service.add_document(chunk, chunk_metadata)
                if success:
                    added_documents += 1
            
            # Add to vector store as well
            await vector_store.add_documents(chunks, [
                {
                    "id": f"{Path(file_path).stem}_chunk_{i}",
                    "source": "uploaded_file",
                    "file_type": file_type,
                    **file_metadata
                }
                for i in range(len(chunks))
            ])
            
            return {
                "success": True,
                "file_path": file_path,
                "chunks_processed": added_documents,
                "total_chunks": len(chunks),
                "characters_extracted": len(extracted_text),
                "metadata": file_metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    async def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from different file types"""
        
        try:
            if file_type == 'txt':
                return await self.extract_from_txt(file_path)
            elif file_type == 'pdf' and PDF_SUPPORT:
                return await self.extract_from_pdf(file_path)
            elif file_type == 'docx' and DOCX_SUPPORT:
                return await self.extract_from_docx(file_path)
            elif file_type in ['xls', 'xlsx'] and EXCEL_SUPPORT:
                return await self.extract_from_excel(file_path)
            else:
                # Try to read as text
                return await self.extract_from_txt(file_path)
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    async def extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            return ""
    
    async def extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting from PDF: {str(e)}")
            return ""
    
    async def extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {str(e)}")
            return ""
    
    async def extract_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            text = ""
            
            for sheet_name, sheet_df in df.items():
                text += f"Sheet: {sheet_name}\n"
                text += sheet_df.to_string(index=False) + "\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting from Excel: {str(e)}")
            return ""
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        
        # Clean the text
        text = self.clean_text(text)
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            # If not at the end of text, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                best_break = end
                
                for ending in sentence_endings:
                    last_occurrence = text.rfind(ending, start, end)
                    if last_occurrence > start + self.chunk_size // 2:
                        best_break = last_occurrence + len(ending)
                        break
                
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/]', '', text)
        
        # Fix common issues
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        return text.strip()
    
    async def process_bulk_upload(
        self, 
        file_paths: List[str], 
        default_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple files in bulk"""
        
        results = []
        
        for file_path in file_paths:
            # Detect file type
            mime_type, _ = mimetypes.guess_type(file_path)
            file_type = self.supported_formats.get(mime_type, 'txt')
            
            # Process file
            result = await self.process_file(file_path, file_type, default_metadata)
            results.append(result)
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        return results
    
    async def create_campus_document_library(self):
        """Create sample campus documents for testing"""
        
        sample_documents = {
            "fee_structure_2024.txt": """
FEE STRUCTURE 2024-25

UNDERGRADUATE PROGRAMS:
- Computer Science Engineering: ₹85,000 per semester
- Electronics & Communication: ₹80,000 per semester  
- Mechanical Engineering: ₹75,000 per semester
- Civil Engineering: ₹70,000 per semester

POSTGRADUATE PROGRAMS:
- M.Tech Computer Science: ₹95,000 per semester
- MBA: ₹1,20,000 per semester

ADDITIONAL FEES:
- Registration Fee: ₹5,000 (one-time)
- Library Fee: ₹2,000 per semester
- Lab Fee: ₹3,000 per semester
- Development Fund: ₹10,000 per year

PAYMENT SCHEDULE:
- First Installment: January 15th
- Second Installment: August 15th
- Late Fee: ₹500 per day after deadline

SCHOLARSHIP DISCOUNTS:
- Merit Scholarship: 25% to 100% waiver
- Need-based Aid: Up to ₹50,000
- Sports Quota: 50% waiver for state players
            """,
            
            "admission_guide_2024.txt": """
ADMISSION GUIDE 2024-25

IMPORTANT DATES:
- Application Start: March 1, 2024
- Application Deadline: May 31, 2024
- Entrance Exam: June 15, 2024
- Result Declaration: July 1, 2024
- Counseling: July 5-20, 2024
- Admission Confirmation: August 5, 2024

ELIGIBILITY CRITERIA:

For B.Tech Programs:
- 12th with Physics, Chemistry, Mathematics
- Minimum 75% marks (70% for reserved categories)
- Valid JEE Main score

For MBA:
- Bachelor's degree in any discipline
- Minimum 50% marks
- Valid CAT/MAT/XAT score

REQUIRED DOCUMENTS:
1. 10th and 12th mark sheets
2. JEE Main scorecard
3. Character certificate
4. Migration certificate
5. Caste certificate (if applicable)
6. Income certificate (for scholarships)
7. Medical fitness certificate

APPLICATION PROCESS:
1. Visit college website
2. Fill online application form
3. Upload required documents
4. Pay application fee: ₹1,500
5. Submit form before deadline

CONTACT INFORMATION:
- Admission Office: +91-9876543210
- Email: admissions@college.edu
- Address: Admission Block, Ground Floor
            """,
            
            "campus_facilities.txt": """
CAMPUS FACILITIES

ACADEMIC FACILITIES:
- Central Library: 2 lakh books, digital library
- Computer Labs: 8 labs with 400+ systems
- Language Lab: Audio-visual learning
- Seminar Halls: 5 halls with 100-300 capacity
- Auditorium: 1000 seating capacity

HOSTEL FACILITIES:
Boys Hostels: 3 blocks, 600 rooms
Girls Hostels: 2 blocks, 400 rooms
- Single/Double/Triple occupancy
- 24/7 security and wifi
- Common rooms with TV
- Study halls and recreation

DINING FACILITIES:
- Main Cafeteria: 500+ seating
- Food Court: 8 different cuisines
- Coffee Shops: 3 locations
- 24/7 canteen in hostels

SPORTS FACILITIES:
- Football Ground: FIFA standard
- Basketball Courts: 4 courts
- Tennis Courts: 2 courts
- Swimming Pool: Olympic size
- Gym: Modern equipment
- Indoor games: TT, Badminton, Chess

HEALTH & WELLNESS:
- Medical Center: 24/7 emergency
- Ambulance Service
- Pharmacy
- Counseling Center
- Yoga & Meditation Center

TRANSPORTATION:
- Campus shuttle service
- Bus connectivity to city
- Parking for 500+ vehicles

OTHER SERVICES:
- ATM: 3 different banks
- Post Office
- Stationary Shop
- Laundry Service
- Internet: 1 Gbps fiber
            """
        }
        
        # Create and process sample documents
        for filename, content in sample_documents.items():
            file_path = os.path.join(self.upload_dir, filename)
            
            # Write content to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Process the file
            await self.process_file(
                file_path, 
                'txt',
                {
                    "category": "campus_info",
                    "source": "official_document",
                    "created_by": "system"
                }
            )
        
        logger.info(f"Created and processed {len(sample_documents)} sample documents")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_formats.values())
    
    def validate_file(self, file_path: str, max_size_mb: int = 10) -> Dict[str, Any]:
        """Validate uploaded file"""
        
        if not os.path.exists(file_path):
            return {"valid": False, "error": "File not found"}
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > max_size_mb * 1024 * 1024:
            return {"valid": False, "error": f"File too large. Maximum size: {max_size_mb}MB"}
        
        # Check file type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type not in self.supported_formats:
            return {
                "valid": False, 
                "error": f"Unsupported file type. Supported: {', '.join(self.get_supported_formats())}"
            }
        
        return {
            "valid": True,
            "file_size": file_size,
            "file_type": self.supported_formats[mime_type],
            "mime_type": mime_type
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics"""
        
        upload_files = list(Path(self.upload_dir).glob("*"))
        processed_files = list(Path(self.processed_dir).glob("*"))
        
        return {
            "uploaded_files": len(upload_files),
            "processed_files": len(processed_files),
            "supported_formats": self.get_supported_formats(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

# Global document processor instance
document_processor = DocumentProcessor()
