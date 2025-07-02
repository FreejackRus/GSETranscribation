# SFTP Configuration
SFTP_HOST = "192.168.33.3"
SFTP_USER = "vmbrowse"
SFTP_PASSWORD = "ShGSE08!"
SFTP_PORT = 22
SFTP_REMOTE_PATH = "."  # Текущая директория на сервере
LOCAL_DOWNLOAD_PATH = "downloaded_audio"
PROCESSED_FILES_LOG = "processed_files.log"

# GLPI Configuration
GLPI_URL = "https://ticket.peremena.ru/apirest.php/"
GLPI_APP_TOKEN = "8guPPJiAQ9rkleijYVjax3UB3UsCqlRMoOGXtIy2"
GLPI_USER_TOKEN = "oG0zFF8H6p98LU63hQ8SjSKq6lEBA7996L1vkKdG"

# Audio Processing
WHISPER_MODEL_PATH = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LLM_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

# Archive
ARCHIVE_DIR = "processed_archive"