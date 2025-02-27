from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1024, chunk_overlap=128):
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
