import os
import warnings
from langchain_community.document_loaders import PyMuPDFLoader
from logger_config import logger

# Suppress warnings
warnings.filterwarnings("ignore")

def load_pdfs_from_directory(directory):
    """Load PDF files from a specified directory and return their contents."""
    logger.info(f"Scanning directory for PDF files: {directory}")
    pdf_paths = []

    # Gather PDF file paths
    for root, dirs, files in os.walk(directory):
        pdf_paths.extend(os.path.join(root, file) for file in files if file.lower().endswith('.pdf'))
    
    if not pdf_paths:
        logger.warning(f"No PDF files found in directory: {directory}")

    logger.info(f"Found {len(pdf_paths)} PDF file(s).")

    # Load documents from each PDF file
    docs = []
    for pdf in pdf_paths:
        logger.info(f"Processing PDF: {pdf}")
        try:
            loader = PyMuPDFLoader(pdf)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            logger.info(f"Successfully loaded {len(loaded_docs)} document(s) from: {pdf}")
        except Exception as e:
            logger.error(f"Error loading {pdf}: {e}", exc_info=True)
    
    logger.info(f"Completed processing. Total documents loaded: {len(docs)}.")
    return docs
