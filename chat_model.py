from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from retriever import configure_retriever
import os

def create_rag_chain(vector_store):
    """Construct the Retrieval-Augmented Generation (RAG) chain."""
    retriever = configure_retriever(vector_store)

    assistant_prompt_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""

    chat_prompt = ChatPromptTemplate.from_template(assistant_prompt_template)
    
    # Using Hugging Face Llama 3.2-3B Instruct
    chat_model = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_TOKEN'),
        temperature=0.1,
        max_new_tokens=512,
        top_k=50,
    )

    return (
        {"context": retriever | format_document_contents, "question": RunnablePassthrough()}
        | chat_prompt
        | chat_model
        | StrOutputParser()
    )

def format_document_contents(documents):
    """Format the contents of retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)