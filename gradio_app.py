"""
Gradio Web Interface for RAG Research Assistant
Deploy with: python gradio_app.py
"""

import gradio as gr
from dotenv import load_dotenv
from document_loader import load_pdfs_from_directory
from text_splitter import split_documents
from vector_store import create_vector_store
from chat_model import create_rag_chain
from retriever import configure_retriever
from lg_langGraph import setup_dspy, run_with_langgraph, run_with_dspy
import time
import json

# Load environment
load_dotenv()

# Initialize system (do this once at startup)
print("🔄 Loading documents and initializing RAG system...")
docs = load_pdfs_from_directory('rag-data')
document_chunks = split_documents(docs)
vector_store = create_vector_store(document_chunks)
retriever = configure_retriever(vector_store)
rag_chain = create_rag_chain(vector_store)
dspy_enabled = setup_dspy()
print("✅ System ready!")

# Track query statistics
query_stats = {
    "total_queries": 0,
    "mode_usage": {"Standard RAG": 0, "LangGraph": 0, "DSPy": 0},
    "avg_latencies": {"Standard RAG": [], "LangGraph": [], "DSPy": []}
}

def process_query(message, mode, history):
    """Process user query with selected RAG mode"""
    
    if not message.strip():
        return history, ""
    
    # Track stats
    query_stats["total_queries"] += 1
    query_stats["mode_usage"][mode] += 1
    
    start_time = time.time()
    
    try:
        # Run selected mode
        if mode == "Standard RAG":
            response = rag_chain.invoke(message)
        elif mode == "LangGraph":
            response = run_with_langgraph(message, retriever, rag_chain)
        elif mode == "DSPy":
            if not dspy_enabled:
                response = "DSPy mode is not available. Please use Standard RAG or LangGraph."
            else:
                response = run_with_dspy(message, retriever)
        
        latency = time.time() - start_time
        query_stats["avg_latencies"][mode].append(latency)
        
        # Get source documents
        docs = retriever.invoke(message)
        sources = list(set([doc.metadata.get('source', 'unknown').split('/')[-1] 
                           for doc in docs[:3]]))
        
        # Format response
        formatted_response = f"{response}\n\n"
        formatted_response += f"**📚 Sources:** {', '.join(sources)}\n"
        formatted_response += f"**⏱️ Latency:** {latency:.2f}s"
        
        # Gradio 4+ expects list of dicts with 'role' and 'content'
        # But actually expects it as the component value, not appending
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": formatted_response}
        ]
        
        return new_history, ""
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        return new_history, ""

def get_stats():
    """Get current statistics"""
    if query_stats["total_queries"] == 0:
        return "No queries yet. Try asking a question!"
    
    stats_text = f"## 📊 Session Statistics\n\n"
    stats_text += f"**Total Queries:** {query_stats['total_queries']}\n\n"
    
    stats_text += "**Mode Usage:**\n"
    for mode, count in query_stats["mode_usage"].items():
        percentage = (count / query_stats["total_queries"] * 100) if query_stats["total_queries"] > 0 else 0
        stats_text += f"- {mode}: {count} ({percentage:.1f}%)\n"
    
    stats_text += "\n**Average Latencies:**\n"
    for mode, latencies in query_stats["avg_latencies"].items():
        if latencies:
            avg = sum(latencies) / len(latencies)
            stats_text += f"- {mode}: {avg:.2f}s\n"
    
    return stats_text

def clear_chat():
    """Clear chat history"""
    return [], ""

# Example questions
example_questions = [
    "What is Proximal Policy Optimization?",
    "How does BERT work?",
    "Explain XGBoost",
    "What is Layer Normalization?",
    "How does StyleGAN generate images?",
    "What is the difference between AdaBoost and XGBoost?"
]

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    title="AI Research Assistant",
    css="""
        .gradio-container {max-width: 1200px !important}
        .chat-bubble {border-radius: 10px; padding: 10px;}
    """
) as demo:
    
    gr.Markdown("""
    # 🤖 AI Research Assistant
    ### Ask questions about 30+ ML/RL research papers
    Powered by LangChain, LangGraph, and DSPy with LLaMA 3.2
    """)
    
    with gr.Row():
        # Main chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="e.g., 'What is Proximal Policy Optimization?'",
                    label="Your Question",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat 🗑️", size="sm")
            
            gr.Markdown("### 💡 Example Questions:")
            with gr.Row():
                for i in range(0, len(example_questions), 2):
                    with gr.Column():
                        if i < len(example_questions):
                            gr.Button(example_questions[i], size="sm").click(
                                lambda x=example_questions[i]: x,
                                outputs=msg
                            )
                        if i+1 < len(example_questions):
                            gr.Button(example_questions[i+1], size="sm").click(
                                lambda x=example_questions[i+1]: x,
                                outputs=msg
                            )
        
        # Sidebar with controls and stats
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            mode = gr.Radio(
                ["Standard RAG", "LangGraph", "DSPy"],
                label="Select RAG Mode",
                value="DSPy",
                info="Choose your retrieval strategy"
            )
            
            gr.Markdown("""
            ### 📈 Performance Metrics
            
            Based on 37 test queries:
            
            | Mode | Quality | Speed | Success |
            |------|---------|-------|---------|
            | **DSPy** ⭐ | 76% | 1.06s | 94% |
            | Standard | 74% | 1.79s | 88% |
            | LangGraph | 63% | 1.31s | 60% |
            
            **Recommendation:** Use DSPy for best results
            """)
            
            gr.Markdown("""
            ### 📚 Indexed Papers
            
            **Machine Learning:**
            - XGBoost, AdaBoost
            - BERT, Layer Normalization
            - StyleGAN, Transfer Learning
            - PCA, LDA, Naive Bayes
            
            **Reinforcement Learning:**
            - PPO (Proximal Policy Optimization)
            - RLHF (Human Feedback)
            - Alignment Research
            
            *Total: 30+ research papers*
            """)
            
            stats_display = gr.Markdown(
                "## 📊 Statistics\nNo queries yet.",
                label="Session Statistics"
            )
            
            refresh_stats_btn = gr.Button("Refresh Stats 🔄", size="sm")
    
    # Event handlers
    submit_btn.click(
        process_query,
        inputs=[msg, mode, chatbot],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        process_query,
        inputs=[msg, mode, chatbot],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )
    
    refresh_stats_btn.click(
        get_stats,
        outputs=stats_display
    )
    
    # Footer
    gr.Markdown("""
    ---
    Built with ❤️ using LangChain, LangGraph, DSPy, and Gradio
    
    [GitHub](#) | [LinkedIn](#) | [Portfolio](#)
    """)

# Launch app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )