from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from config import CHAT_MODEL, BASE_URL
from retriever import configure_retriever

def create_rag_chain(vector_store):
    """Construct the Retrieval-Augmented Generation (RAG) chain."""
    retriever = configure_retriever(vector_store)

    assistant_prompt_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>

    Question: {question}

    Context: {context}

    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    chat_prompt = ChatPromptTemplate.from_template(assistant_prompt_template)
    chat_model = ChatOllama(model=CHAT_MODEL, base_url=BASE_URL)

    return (
        {"context": retriever | format_document_contents, "question": RunnablePassthrough()}
        | chat_prompt
        | chat_model
        | StrOutputParser()
    )

def format_document_contents(documents):
    """Format the contents of retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)
