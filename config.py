import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# SFTP Configuration
SFTP_HOST = os.getenv("SFTP_HOST", "192.168.33.3")
SFTP_USER = os.getenv("SFTP_USER", "vmbrowse")
SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")
SFTP_PORT = int(os.getenv("SFTP_PORT", "22"))
SFTP_REMOTE_PATH = "."  
LOCAL_DOWNLOAD_PATH = "downloaded_audio"
PROCESSED_FILES_LOG = "processed_files.log"

# GLPI Configuration
GLPI_URL = os.getenv("GLPI_URL", "https://ticket.peremena.ru/apirest.php/")
GLPI_APP_TOKEN = os.getenv("GLPI_APP_TOKEN")
GLPI_USER_TOKEN = os.getenv("GLPI_USER_TOKEN")

# Audio Processing
WHISPER_MODEL_PATH = "large"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LLM_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

# Archive
ARCHIVE_DIR = "processed_archive"