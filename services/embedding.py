import random

class EmbeddingService():
    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        # Seed based on input so it's "deterministic"
        if not text:
            return ValueError("Input text cannot be empty")
        seed = abs(hash(text)) % 10000
        random.seed(seed)

        embedding = [random.random() for _ in range(self._dimension)]
        return embedding
    
    def get_dimension(self) -> int:
        return self._dimension
    
if __name__ == "__main__":
    service = EmbeddingService()

    text = "This is test document"
    embedding = service.embed(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {service.get_dimension()}")
    print(f"Embedding (first 5 values): {embedding[:5]}")