"""
Simple integration of LangGraph and DSPy with your existing RAG system
Just add this file to your project and update main.py
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
import operator
import dspy

# ============================================
# LANGGRAPH PART
# ============================================

class GraphState(TypedDict):
    """State for the RAG workflow"""
    question: str
    generation: str
    documents: List[Document]
    needs_better_docs: bool

def retrieve_documents(state: GraphState, retriever) -> GraphState:
    """Retrieve documents from vector store"""
    question = state["question"]
    documents = retriever.invoke(question)
    
    return {
        **state,
        "documents": documents
    }

def grade_documents(state: GraphState) -> GraphState:
    """Grade if documents are relevant"""
    question = state["question"]
    documents = state["documents"]
    
    # Simple relevance check - are key question words in docs?
    question_words = set(question.lower().split()[:5])
    filtered_docs = []
    
    for doc in documents:
        doc_words = set(doc.page_content.lower().split())
        # If at least 2 question words appear in doc, keep it
        if len(question_words & doc_words) >= 2:
            filtered_docs.append(doc)
    
    return {
        **state,
        "documents": filtered_docs,
        "needs_better_docs": len(filtered_docs) == 0
    }

def generate_answer(state: GraphState, rag_chain) -> GraphState:
    """Generate answer using the documents"""
    question = state["question"]
    generation = rag_chain.invoke(question)
    
    return {
        **state,
        "generation": generation
    }

def no_docs_fallback(state: GraphState) -> GraphState:
    """Handle case when no good documents found"""
    return {
        **state,
        "generation": "I don't have enough relevant information to answer this question accurately."
    }

def decide_next_step(state: GraphState) -> str:
    """Decide whether to generate or return fallback"""
    if state.get("needs_better_docs"):
        return "fallback"
    return "generate"

def build_rag_workflow(retriever, rag_chain):
    """Build the LangGraph workflow"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", lambda s: retrieve_documents(s, retriever))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", lambda s: generate_answer(s, rag_chain))
    workflow.add_node("fallback", no_docs_fallback)
    
    # Build the graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_step,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)
    
    return workflow.compile()

# ============================================
# DSPY PART - Simplified without structured outputs
# ============================================

class SimpleRAG(dspy.Module):
    """Simplified DSPy RAG - uses direct prompting"""
    
    def __init__(self, k=3):
        super().__init__()
        self.k = k
    
    def forward(self, question, context_docs):
        # Format context from documents
        if not context_docs:
            return "I don't have enough information to answer this question."
        
        context = "\n\n".join(doc.page_content[:500] for doc in context_docs[:self.k])
        
        # Create a simple prompt without structured output
        prompt = f"""Based on the following context, answer the question concisely in 2-3 sentences.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            # Get the configured LM
            lm = dspy.settings.lm
            # Call it directly
            response = lm.basic_request(prompt)
            return response.strip()
        except Exception as e:
            print(f"DSPy prediction failed: {e}")
            return f"Error generating answer with DSPy"

def setup_dspy():
    """Initialize DSPy with Ollama"""
    try:
        import os
        from langchain_ollama import ChatOllama
        
        model_name = os.getenv('CHAT_MODEL', 'llama3.2:1b')
        base_url = os.getenv('BASE_URL', 'http://localhost:11434')
        
        # Properly inherit from BaseLM with all required methods
        class OllamaLM(dspy.BaseLM):
            def __init__(self, model, base_url):
                # Call parent with model name
                super().__init__(model=model)
                self.llm = ChatOllama(model=model, base_url=base_url)
            
            def basic_request(self, prompt, **kwargs):
                """This is the main method DSPy calls"""
                try:
                    response = self.llm.invoke(prompt)
                    return response.content.strip()
                except Exception as e:
                    print(f"Error in basic_request: {e}")
                    raise
            
            def __call__(self, prompt=None, messages=None, **kwargs):
                """Handle both prompt and messages format"""
                if messages:
                    # Convert messages to prompt
                    prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
                
                if prompt:
                    return self.basic_request(prompt, **kwargs)
                
                raise ValueError("Either prompt or messages must be provided")
        
        lm = OllamaLM(model_name, base_url)
        dspy.settings.configure(lm=lm)
        print(f" DSPy configured with {model_name}")
        return True
            
    except Exception as e:
        print(f" DSPy setup failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Mode 3 will be disabled")
        return False

# ============================================
# INTEGRATION FUNCTIONS
# ============================================

def run_with_langgraph(question: str, retriever, rag_chain):
    """Run query using LangGraph workflow"""
    workflow = build_rag_workflow(retriever, rag_chain)
    
    result = workflow.invoke({
        "question": question,
        "generation": "",
        "documents": [],
        "needs_better_docs": False
    })
    
    return result["generation"]

def run_with_dspy(question: str, retriever):
    """Run query using DSPy optimization"""
    # Get documents
    docs = retriever.invoke(question)
    
    # Use DSPy module
    rag_module = SimpleRAG(k=3)
    answer = rag_module(question, docs)
    
    return answer