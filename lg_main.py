"""
This replaces the main() function and everything after it in your current file.
Keep everything BEFORE the main() function (all your imports and other functions).
"""

from dotenv import load_dotenv
from document_loader import load_pdfs_from_directory
from text_splitter import split_documents
from vector_store import create_vector_store
from chat_model import create_rag_chain
from retriever import configure_retriever
from logger_config import logger
import sys

# ADD THIS NEW IMPORT at the top of your file
from lg_langGraph import (
    setup_dspy,
    run_with_langgraph,
    run_with_dspy
)

def main():
    # Load environment variables
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

    # Create retriever and RAG chain
    try:
        logger.info("Initializing retriever and RAG chain...")
        retriever = configure_retriever(vector_store)
        rag_chain = create_rag_chain(vector_store)
        logger.info("Retriever and RAG chain initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize retriever.", exc_info=True)
        sys.exit(1)

    # Setup DSPy
    dspy_enabled = setup_dspy()
    if dspy_enabled:
        print("DSPy optimization enabled")
    else:
        print("Failed! DSPy optimization disabled")

    # Start chatbot with mode selection
    print("\n" + "="*60)
    print("Advanced RAG Chatbot")
    print("="*60)
    print("\nAvailable modes:")
    print("  1. Standard RAG (original)")
    print("  2. LangGraph workflow (with document grading)")
    print("  3. DSPy optimized (automatic prompt optimization)")
    print("\nCommands:")
    print("  'mode 1/2/3' - Switch modes")
    print("  'exit' - Quit")
    print("="*60 + "\n")
    
    current_mode = 1
    mode_names = {1: "Standard RAG", 2: "LangGraph", 3: "DSPy"}
    
    while True:
        try:
            # Show current mode
            print(f"[{mode_names[current_mode]}] ", end="")
            user_input = input("You: ")
            
            # Handle commands
            if user_input.lower() == 'exit':
                logger.info("User exited the chatbot.")
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('mode '):
                try:
                    new_mode = int(user_input.split()[1])
                    if new_mode in [1, 2, 3]:
                        if new_mode == 3 and not dspy_enabled:
                            print("DSPy is not available. Staying in current mode.")
                        else:
                            current_mode = new_mode
                            print(f"Switched to {mode_names[current_mode]} mode\n")
                    else:
                        print("Invalid mode. Choose 1, 2, or 3\n")
                except:
                    print("Invalid command. Use 'mode 1', 'mode 2', or 'mode 3'\n")
                continue
            
            # Process query based on current mode
            if current_mode == 1:
                # Standard RAG (original)
                response = rag_chain.invoke(user_input)
                
            elif current_mode == 2:
                # LangGraph workflow
                response = run_with_langgraph(user_input, retriever, rag_chain)
                
            elif current_mode == 3:
                # DSPy optimized
                response = run_with_dspy(user_input, retriever)
            
            logger.info("Processed user question successfully.")
            print(f"Assistant: {response}\n")
            
        except Exception as e:
            logger.error("An error occurred during chatbot interaction.", exc_info=True)
            print(f"An error occurred: {e}\n")
            print("Please try again.\n")


if __name__ == "__main__":
    main()