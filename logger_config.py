import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/application.log"),  # Shared log file
        # logging.StreamHandler()  # Optional: Keep console output
    ]
)

# Retrieve the shared logger
logger = logging.getLogger("shared_logger")