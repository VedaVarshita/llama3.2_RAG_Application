from dotenv import load_dotenv
from document_loader import load_pdfs_from_directory
from text_splitter import split_documents
from vector_store import create_vector_store
from chat_model import create_rag_chain
from logger_config import logger
import sys

def main():
    # Load environment variables from .env file
    try:
        logger.info("Loading environment variables...")
        load_dotenv()
        logger.info("Environment variables loaded successfully.")
    except Exception as e:
        logger.error("Failed to load environment variables.", exc_info=True)
        sys.exit(1)

    # Load documents
    try:
        logger.info("Starting to load documents from the directory...")
        docs = load_pdfs_from_directory('rag-data')
        logger.info(f"Successfully loaded {len(docs)} documents.")
    except Exception as e:
        logger.error("Failed to load documents.", exc_info=True)
        sys.exit(1)

    # Split documents into chunks
    try:
        logger.info("Splitting documents into smaller chunks...")
        document_chunks = split_documents(docs)
        logger.info(f"Split documents into {len(document_chunks)} chunks.")
    except Exception as e:
        logger.error("Failed to split documents.", exc_info=True)
        sys.exit(1)

    # Create vector store
    try:
        logger.info("Creating the vector store...")
        vector_store = create_vector_store(document_chunks)
        logger.info("Vector store created successfully.")
    except Exception as e:
        logger.error("Failed to create vector store.", exc_info=True)
        sys.exit(1)

    # Create RAG chain
    try:
        logger.info("Initializing the RAG chain...")
        rag_chain = create_rag_chain(vector_store)
        logger.info("RAG chain initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize the RAG chain.", exc_info=True)
        sys.exit(1)

    # Start chatbot interaction
    print("Hello world!, Type 'exit' to end the conversation.")
    
    while True:
        try:
            # Get user input
            user_question = input("You: ")
            
            # Check if user wants to exit
            if user_question.lower() == 'exit':
                logger.info("User exited the chatbot.")
                print("Byee :)")
                break

            # Fetch the answer
            response = rag_chain.invoke(user_question)
            logger.info(f"Processed user question successfully.")
            
            # Print the final output
            print("Assistant:", response)
            # print("¯\_(ツ)_/¯:", response)
        except Exception as e:
            logger.error("An error occurred during chatbot interaction.", exc_info=True)
            print("An error occurred. Please try again.")




if __name__ == "__main__":
    main()
