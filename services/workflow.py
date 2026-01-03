from typing import Dict, Any, List
from langgraph.graph import StateGraph, END


class RagWorkflow:    
    def __init__(self, document_store, embedding_service):
        
        self._document_store = document_store
        self._embedding_service = embedding_service
        self._chain = self._build_graph()
    
    def _retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        
        # Generate embedding for the question
        query_embedding = self._embedding_service.embed(question)
        
        # Search for relevant documents
        results = self._document_store.search(query_embedding, limit=2)
        
        # Update state with context
        state["context"] = results
        return state
    
    def _answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context = state.get("context", [])
        
        if context:
            answer = f"I found this: '{context[0][:100]}...'"
        else:
            answer = "Sorry, I don't know."
        
        state["answer"] = answer
        return state
    
    def _build_graph(self) -> Any:
        # Create state graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("answer", self._answer)
        
        # Define flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        
        # Compile and return
        return workflow.compile()
    
    def execute(self, question: str) -> Dict[str, Any]:
        if not question:
            raise ValueError("Question cannot be empty or None")
        
        # Initialize state
        initial_state = {"question": question}
        
        # Run the workflow
        result = self._chain.invoke(initial_state)
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": result.get("context", [])
        }
    
if __name__ == "__main__":
    from embedding import EmbeddingService
    from storage import create_document_store

    # Initialize services
    embedding_service = EmbeddingService(dimension=128)
    document_store, _ = create_document_store(vector_size=128)

    # Add sample documents
    docs = [
        "The capital of France is Paris.",
        "The largest planet in our solar system is Jupiter.",
        "The Python programming language was created by Guido van Rossum."
    ]
    for idx, doc in enumerate(docs):
        embedding = embedding_service.embed(doc)
        document_store.add_document(doc_id=idx, text=doc, embedding=embedding)

    # Initialize RAG workflow
    rag_workflow = RagWorkflow(document_store, embedding_service)

    # Execute a sample question
    question = "What is the capital of France?"
    result = rag_workflow.execute(question)

    print("Question:", result["question"])
    print("Answer:", result["answer"])
    print("Context Used:", result["context"])