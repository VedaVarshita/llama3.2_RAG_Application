import os

# Configuration constants
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
# CHAT_MODEL = os.getenv('CHAT_MODEL', 'llama3.2:1b')
# BASE_URL = os.getenv('BASE_URL', 'http://localhost:11434')



CHAT_MODEL = "gemini-pro"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_URL = None  # Not used with Gemini