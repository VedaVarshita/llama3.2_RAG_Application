import os

# Configuration constants
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
CHAT_MODEL = os.getenv('CHAT_MODEL', 'llama3.2:1b')
BASE_URL = os.getenv('BASE_URL', 'http://localhost:11434')
