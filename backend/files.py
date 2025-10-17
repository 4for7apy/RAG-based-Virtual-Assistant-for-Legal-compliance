"""
Files Router - Document Upload and Processing for RAG
"""

import os
import tempfile
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional
import logging

from app.services.document_processor import document_processor
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: Optional[str] = "general",
    description: Optional[str] = None
):
    """Upload and process document for RAG system"""
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Validate the uploaded file
        validation = document_processor.validate_file(tmp_path)
        if not validation["valid"]:
            os.unlink(tmp_path)  # Clean up
            raise HTTPException(status_code=400, detail=validation["error"])
        
        # Process the file in background
        metadata = {
            "filename": file.filename,
            "category": category,
            "description": description,
            "content_type": file.content_type,
            "uploaded_by": "user"  # Would be actual user in real app
        }
        
        background_tasks.add_task(
            process_uploaded_file,
            tmp_path,
            validation["file_type"],
            metadata
        )
        
        return {
            "message": "File uploaded successfully and processing started",
            "filename": file.filename,
            "file_size": validation["file_size"],
            "file_type": validation["file_type"],
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_uploaded_file(file_path: str, file_type: str, metadata: dict):
    """Background task to process uploaded file"""
    try:
        result = await document_processor.process_file(file_path, file_type, metadata)
        
        if result["success"]:
            logger.info(f"Successfully processed {metadata['filename']}: {result['chunks_processed']} chunks")
        else:
            logger.error(f"Failed to process {metadata['filename']}: {result['error']}")
        
        # Clean up temporary file
        os.unlink(file_path)
        
    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)

@router.post("/upload-multiple")
async def upload_multiple_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    category: Optional[str] = "general"
):
    """Upload multiple files for batch processing"""
    
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
        
        upload_results = []
        temp_files = []
        
        for file in files:
            if not file.filename:
                continue
                
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
                temp_files.append(tmp_path)
            
            # Validate file
            validation = document_processor.validate_file(tmp_path)
            
            if validation["valid"]:
                upload_results.append({
                    "filename": file.filename,
                    "status": "queued",
                    "file_size": validation["file_size"],
                    "file_type": validation["file_type"]
                })
            else:
                upload_results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": validation["error"]
                })
                # Remove invalid file
                os.unlink(tmp_path)
                temp_files.remove(tmp_path)
        
        # Process valid files in background
        if temp_files:
            background_tasks.add_task(
                process_multiple_files,
                temp_files,
                category
            )
        
        return {
            "message": f"Uploaded {len(temp_files)} files for processing",
            "results": upload_results
        }
        
    except Exception as e:
        logger.error(f"Error uploading multiple files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")

async def process_multiple_files(file_paths: List[str], category: str):
    """Background task to process multiple files"""
    try:
        default_metadata = {
            "category": category,
            "uploaded_by": "user",
            "batch_upload": True
        }
        
        # Determine file types
        file_info = []
        for path in file_paths:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(path)
            file_type = document_processor.supported_formats.get(mime_type, 'txt')
            file_info.append((path, file_type))
        
        # Process files
        results = []
        for file_path, file_type in file_info:
            result = await document_processor.process_file(file_path, file_type, default_metadata)
            results.append(result)
            
            # Clean up
            os.unlink(file_path)
        
        logger.info(f"Batch processing completed: {len(results)} files processed")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        # Clean up remaining files
        for path in file_paths:
            if os.path.exists(path):
                os.unlink(path)

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": document_processor.get_supported_formats(),
        "max_file_size": "10MB",
        "description": "Upload campus documents to enhance the chatbot's knowledge"
    }

@router.get("/processing-stats")
async def get_processing_stats():
    """Get document processing statistics"""
    try:
        doc_stats = document_processor.get_processing_stats()
        rag_stats = rag_service.get_statistics()
        
        return {
            "document_processing": doc_stats,
            "rag_system": rag_stats,
            "total_knowledge_base_size": rag_stats["total_documents"]
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@router.post("/create-sample-data")
async def create_sample_data():
    """Create sample campus documents for testing"""
    try:
        await document_processor.create_campus_document_library()
        return {
            "message": "Sample campus documents created and processed successfully",
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create sample data")

@router.delete("/clear-knowledge-base")
async def clear_knowledge_base():
    """Clear the entire knowledge base (admin only)"""
    try:
        # This would typically require admin authentication
        # For demo purposes, allowing direct access
        
        # Clear RAG service documents
        rag_service.documents.clear()
        rag_service.metadata.clear()
        rag_service.is_trained = False
        
        # Clear vector store
        from app.services.advanced_embeddings import vector_store
        vector_store.vectors.clear()
        vector_store.documents.clear()
        vector_store.metadata.clear()
        vector_store.index_map.clear()
        
        return {
            "message": "Knowledge base cleared successfully",
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
