from typing import List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The question to ask the RAG system",
    )


class DocumentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The document text to add to the knowledge base",
    )


class QuestionResponse(BaseModel):
    question: str = Field(
        ...,
        description="The original question"
    )
    answer: str = Field(
        ...,
        description="The generated answer"
    )
    context_used: List[str] = Field(
        default_factory=list,
        description="Documents retrieved and used for answering"
    )
    latency_sec: float = Field(
        ...,
        description="Processing time in seconds"
    )


class DocumentResponse(BaseModel):
    id: int = Field(
        ...,
        description="The unique identifier assigned to the document"
    )
    status: str = Field(
        ...,
        description="Status of the operation",
        examples=["added"]
    )


class StatusResponse(BaseModel):
    qdrant_ready: bool = Field(
        ...,
        description="Whether Qdrant vector database is available"
    )
    in_memory_docs_count: int = Field(
        ...,
        description="Number of documents in in-memory fallback storage"
    )
    graph_ready: bool = Field(
        ...,
        description="Whether the RAG workflow is initialized"
    )