from sftp_handler import SFTPAudioProcessor
from logger import setup_logger

setup_logger()

def test_processing():
    print("=== ТЕСТОВЫЙ ЗАПУСК ===")
    processor = SFTPAudioProcessor()
    processor.process_new_files()
    print("=== ТЕСТ ЗАВЕРШЕН ===")

if __name__ == "__ma    in__":
    test_processing()