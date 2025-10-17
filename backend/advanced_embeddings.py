"""
Advanced Embeddings Service
Support for multiple embedding models and vector operations
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json
import hashlib
from datetime import datetime

# Sentence Transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# OpenAI for embeddings (if API key available)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Fallback TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

logger = logging.getLogger(__name__)

class AdvancedEmbeddingsService:
    """Advanced embeddings with multiple model support"""
    
    def __init__(self):
        self.models = {}
        self.current_model = "tfidf"  # Default fallback
        self.cache = {}
        self.vector_dim = 384  # Default dimension
        
    async def initialize(self):
        """Initialize embedding models"""
        try:
            logger.info("Initializing embedding models...")
            
            # Try to load Sentence Transformers
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                await self.load_sentence_transformer()
            
            # Setup OpenAI if available
            if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
                await self.setup_openai_embeddings()
            
            # Always have TF-IDF as fallback
            await self.setup_tfidf_embeddings()
            
            logger.info(f"Embedding service initialized with model: {self.current_model}")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            await self.setup_tfidf_embeddings()  # Fallback
    
    async def load_sentence_transformer(self):
        """Load Sentence Transformer model"""
        try:
            # Use a lightweight multilingual model
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            self.models["sentence_transformer"] = SentenceTransformer(model_name)
            self.current_model = "sentence_transformer"
            self.vector_dim = 384
            logger.info(f"Loaded Sentence Transformer: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load Sentence Transformer: {str(e)}")
    
    async def setup_openai_embeddings(self):
        """Setup OpenAI embeddings"""
        try:
            openai.api_key = settings.OPENAI_API_KEY
            
            # Test the connection
            test_response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input="test"
            )
            
            self.models["openai"] = "text-embedding-ada-002"
            self.current_model = "openai"
            self.vector_dim = 1536
            logger.info("OpenAI embeddings configured")
            
        except Exception as e:
            logger.warning(f"Failed to setup OpenAI embeddings: {str(e)}")
    
    async def setup_tfidf_embeddings(self):
        """Setup TF-IDF as fallback"""
        try:
            self.models["tfidf"] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            if self.current_model not in self.models:
                self.current_model = "tfidf"
                self.vector_dim = 1000
            
            logger.info("TF-IDF embeddings configured")
            
        except Exception as e:
            logger.error(f"Failed to setup TF-IDF: {str(e)}")
    
    async def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None
    ) -> np.ndarray:
        """Get embeddings for text(s)"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        model = model or self.current_model
        
        # Check cache first
        cache_key = self.get_cache_key(texts, model)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if model == "openai" and "openai" in self.models:
                embeddings = await self.get_openai_embeddings(texts)
            elif model == "sentence_transformer" and "sentence_transformer" in self.models:
                embeddings = await self.get_sentence_transformer_embeddings(texts)
            else:
                embeddings = await self.get_tfidf_embeddings(texts)
            
            # Cache the result
            self.cache[cache_key] = embeddings
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            # Fallback to TF-IDF
            return await self.get_tfidf_embeddings(texts)
    
    async def get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get OpenAI embeddings"""
        try:
            response = await openai.Embedding.acreate(
                model=self.models["openai"],
                input=texts
            )
            
            embeddings = [item['embedding'] for item in response['data']]
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    async def get_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get Sentence Transformer embeddings"""
        try:
            model = self.models["sentence_transformer"]
            embeddings = model.encode(texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"Sentence Transformer error: {str(e)}")
            raise
    
    async def get_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get TF-IDF embeddings"""
        try:
            model = self.models["tfidf"]
            
            # If model is not fitted, fit it with the texts
            if not hasattr(model, 'vocabulary_'):
                model.fit(texts)
            
            embeddings = model.transform(texts)
            return embeddings.toarray()
            
        except Exception as e:
            logger.error(f"TF-IDF error: {str(e)}")
            raise
    
    def calculate_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Ensure embeddings are 2D
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def get_cache_key(self, texts: List[str], model: str) -> str:
        """Generate cache key for embeddings"""
        content = f"{model}:{':'.join(texts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        model: Optional[str] = None
    ) -> np.ndarray:
        """Process embeddings in batches for large datasets"""
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.get_embeddings(batch, model)
            all_embeddings.append(batch_embeddings)
            
            # Small delay to prevent rate limiting
            if model == "openai":
                await asyncio.sleep(0.1)
        
        return np.vstack(all_embeddings)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "current_model": self.current_model,
            "available_models": list(self.models.keys()),
            "vector_dimension": self.vector_dim,
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")

# Enhanced Vector Store
class VectorStore:
    """In-memory vector store with persistence"""
    
    def __init__(self, embeddings_service: AdvancedEmbeddingsService):
        self.embeddings_service = embeddings_service
        self.vectors = []
        self.documents = []
        self.metadata = []
        self.index_map = {}
    
    async def add_documents(
        self, 
        documents: List[str], 
        metadata_list: List[Dict[str, Any]]
    ):
        """Add documents to vector store"""
        try:
            # Generate embeddings
            embeddings = await self.embeddings_service.get_embeddings(documents)
            
            # Add to store
            start_idx = len(self.vectors)
            for i, (doc, meta, embedding) in enumerate(zip(documents, metadata_list, embeddings)):
                self.documents.append(doc)
                self.metadata.append(meta)
                self.vectors.append(embedding)
                
                # Create index mapping
                doc_id = meta.get('id', f"doc_{start_idx + i}")
                self.index_map[doc_id] = start_idx + i
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not self.vectors:
                return []
            
            # Get query embedding
            query_embedding = await self.embeddings_service.get_embeddings([query])
            query_vector = query_embedding[0]
            
            # Calculate similarities
            similarities = []
            for i, doc_vector in enumerate(self.vectors):
                similarity = self.embeddings_service.calculate_similarity(
                    query_vector, doc_vector
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            results = []
            for i, (doc_idx, similarity) in enumerate(similarities[:top_k]):
                if similarity > similarity_threshold:
                    results.append({
                        "document": self.documents[doc_idx],
                        "metadata": self.metadata[doc_idx],
                        "similarity": similarity,
                        "rank": i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        if doc_id in self.index_map:
            idx = self.index_map[doc_id]
            return {
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "vector": self.vectors[idx]
            }
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "vector_dimension": len(self.vectors[0]) if self.vectors else 0,
            "embeddings_model": self.embeddings_service.current_model
        }

# Global instances
embeddings_service = AdvancedEmbeddingsService()
vector_store = VectorStore(embeddings_service)
