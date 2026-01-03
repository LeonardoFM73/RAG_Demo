import time
from fastapi import FastAPI, HTTPException
from services.embedding import EmbeddingService
from services.storage import create_document_store, InMemoryDocumentStore
from services.workflow import RagWorkflow
from models.schemas import (
    QuestionRequest,
    DocumentRequest,
    QuestionResponse,
    DocumentResponse,
    StatusResponse
)


class RagApplication:    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "demo_collection",
        embedding_dimension: int = 128
    ):
        # Initialize embedding service
        self.embedding_service = EmbeddingService(dimension=embedding_dimension)
        
        # Initialize document store (Qdrant or in-memory fallback)
        self.document_store, self.is_using_qdrant = create_document_store(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            vector_size=embedding_dimension
        )
        
        # Initialize RAG workflow
        self.workflow = RagWorkflow(
            document_store=self.document_store,
            embedding_service=self.embedding_service
        )
        
        # Document ID counter (simple implementation)
        self.doc_counter = 0
    
    def add_document(self, text: str) -> dict:
        if not text:
            raise ValueError("Document text cannot be empty")
        
        # Generate embedding
        embedding = self.embedding_service.embed(text)
        
        # Get document ID
        doc_id = self.doc_counter
        self.doc_counter += 1
        
        # Store document
        self.document_store.add_document(doc_id, text, embedding)
        
        return {"id": doc_id, "status": "added"}
    
    def ask_question(self, question: str) -> dict:
        if not question:
            raise ValueError("Question cannot be empty")
        
        start_time = time.time()
        
        # Execute workflow
        result = self.workflow.execute(question)
        
        # Calculate latency
        latency = round(time.time() - start_time, 3)
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context_used": result["context"],
            "latency_sec": latency
        }
    
    def get_status(self) -> dict:
        # Get document count for in-memory store
        in_memory_count = 0
        if isinstance(self.document_store, InMemoryDocumentStore):
            in_memory_count = self.document_store.get_document_count()
        
        return {
            "qdrant_ready": self.is_using_qdrant,
            "in_memory_docs_count": in_memory_count,
            "graph_ready": self.workflow is not None
        }


# Initialize FastAPI app
app = FastAPI(
    title="Learning RAG Demo",
    description="A clean, maintainable RAG system using FastAPI and LangGraph",
    version="2.0.0"
)

# Initialize RAG application
rag_app = RagApplication()


# API Endpoints
@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest) -> QuestionResponse:
    try:
        result = rag_app.ask_question(request.question)
        return QuestionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/add", response_model=DocumentResponse)
def add_document(request: DocumentRequest) -> DocumentResponse:
    try:
        result = rag_app.add_document(request.text)
        return DocumentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    result = rag_app.get_status()
    return StatusResponse(**result)


# Health check endpoint
@app.get("/")
def root():
    """Root endpoint for health check."""
    return {
        "message": "Learning RAG Demo API",
        "status": "running",
        "version": "2.0.0"
    }