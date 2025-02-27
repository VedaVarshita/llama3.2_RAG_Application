
def configure_retriever(vector_store):
    """Configure the retriever for the vector store."""
    return vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={
            'k': 3,
            'fetch_k': 100,
            'lambda_mult': 1
        }
    )
