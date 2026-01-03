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