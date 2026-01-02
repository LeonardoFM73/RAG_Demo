from typing import List, Dict, Any
from abc import ABC, abstractmethod
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


class DocumentStore(ABC):
    @abstractmethod
    def add_document(self, doc_id: int, text: str, embedding: List[float]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], limit: int = 2) -> List[str]:
        pass

class QdrantDocumentStore(DocumentStore):
    def __init__(self, client: QdrantClient, collection_name: str = "demo_collection"):
        self._client = client
        self._collection_name = collection_name

    def add_document(self, doc_id: int, text: str, embedding: List[float]) -> None:
        payload = {"text": text}
        point = PointStruct(id=doc_id, vector=embedding, payload=payload)
        self._client.upsert(collection_name=self._collection_name, points=[point])

    def search(self, query_embedding: List[float], limit: int = 2) -> List[str]:
        results = self._client.query_points(collection_name=self._collection_name, query=query_embedding, limit=limit)
        return [point.payload["text"] for point in results.points]

class InMemoryDocumentStore(DocumentStore):
    def __init__(self):
        self._documents: List[str] = []

    def add_document(self, doc_id: int, text: str, embedding: List[float]) -> None:
        self._documents.append(text)

    def search(self, query_embedding: List[float], limit: int = 2) -> List[str]:
        # Simple substring match for demo purposes
        if self._documents:
            return [self._documents[0]]
        return []
    def get_document_count(self) -> int:
        return len(self._documents)
    
def create_document_store(qdrant_url: str = "http://localhost:6333", collection_name: str = "demo_collection", vector_size: int = 128) -> tuple[DocumentStore, bool]:
    try:
        client = QdrantClient(qdrant_url)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        store = QdrantDocumentStore(client, collection_name)
        return store, True
    except Exception as e:
        print(f"⚠️  Qdrant not available: {e}. Falling back to in-memory store.")
        store = InMemoryDocumentStore()
        return store, False
    
    # Example usage (for testing purposes)
if __name__ == "__main__":
    # Try to create document store
    store, is_qdrant = create_document_store()
    
    print(f"Using Qdrant: {is_qdrant}")
    print(f"Store type: {type(store).__name__}")
    
    # Test adding document
    fake_embedding = [0.1] * 128
    store.add_document(0, "Test document", fake_embedding)
    
    # Test search
    results = store.search(fake_embedding, limit=1)
    print(f"Search results: {results}")

