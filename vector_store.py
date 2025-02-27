import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL, BASE_URL

def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks using the specified embedding model."""
    embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=BASE_URL)
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
