import logging

import schedule
import time
from sftp_handler import SFTPAudioProcessor
from logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def processing_job():
    logger.info("=== Starting audio files processing ===")
    processor = SFTPAudioProcessor()
    processor.process_new_files()
    logger.info("=== Processing completed ===\n")


# Настройка расписания
schedule.every().day.at("04:00").do(processing_job)  # Основная ночная обработка

if __name__ == "__main__":
    logger.info("Audio processing service started")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Service stopped")
