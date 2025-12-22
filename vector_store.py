import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks using HuggingFace embeddings."""
    
    # Using sentence-transformers instead of Ollama embeddings
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    sample_vector = embeddings_model.embed_query("this is sample text")
    index = faiss.IndexFlatL2(len(sample_vector))

    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=chunks)
    return vector_store