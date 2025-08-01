import logging
import os
from datetime import datetime

def setup_logger():
    # Создаем директорию для логов если её нет
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Имя файла лога с датой
    log_filename = os.path.join(log_dir, f"gse_transcription_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Вывод в консоль
            logging.FileHandler(log_filename, encoding='utf-8')  # Запись в файл
        ]
    )