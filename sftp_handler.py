import paramiko
import os
from datetime import datetime
import logging
from config import *
from logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


class SFTPAudioProcessor:
    def __init__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.sftp = None
        self.processed_files = self.load_processed_files()

    def connect(self):
        try:
            self.ssh.connect(SFTP_HOST, port=SFTP_PORT,
                             username=SFTP_USER, password=SFTP_PASSWORD)
            self.sftp = self.ssh.open_sftp()
            logger.info("SFTP connection established")
            return True
        except Exception as e:
            logger.error(f"SFTP connection error: {e}")
            return False

    def load_processed_files(self):
        try:
            with open(PROCESSED_FILES_LOG, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            return set()

    def save_processed_file(self, filename):
        with open(PROCESSED_FILES_LOG, 'a') as f:
            f.write(f"{filename}\n")
        self.processed_files.add(filename)

    # В методе get_new_audio_files добавьте проверку на наличие соответствующих txt файлов
    def get_new_audio_files(self):
        try:
            files = self.sftp.listdir(SFTP_REMOTE_PATH)
            wav_files = [f for f in files
                         if f.startswith('msg')
                         and f.endswith('.wav')
                         and f not in self.processed_files
                         and f.replace('.wav', '.txt') in files]  # Проверяем наличие txt файла
            return sorted(wav_files)
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    # Добавьте новый метод для чтения данных из txt файла
    def read_metadata_file(self, filename):
        txt_filename = filename.replace('.wav', '.txt')
        try:
            with self.sftp.open(f"{SFTP_REMOTE_PATH}/{txt_filename}") as f:
                content = f.read().decode('utf-8')
                metadata = {}
                for line in content.splitlines():
                    if 'callerid=' in line:
                        metadata['callerid'] = line.split('callerid="')[1].split('"')[0]
                    elif 'origtime=' in line:
                        metadata['origtime'] = int(line.split('origtime=')[1].strip())
                return metadata
        except Exception as e:
            logger.error(f"Error reading metadata file {txt_filename}: {e}")
            return None

    def download_audio_file(self, filename):
        local_path = os.path.join(LOCAL_DOWNLOAD_PATH, filename)
        try:
            os.makedirs(LOCAL_DOWNLOAD_PATH, exist_ok=True)
            remote_path = f"{SFTP_REMOTE_PATH}/{filename}"
            self.sftp.get(remote_path, local_path)
            logger.info(f"Downloaded: {filename}")
            return local_path
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            return None

    def archive_processed_file(self, filename):
        try:
            archive_dir = f"{SFTP_REMOTE_PATH}/{ARCHIVE_DIR}"
            try:
                self.sftp.mkdir(archive_dir)
            except IOError:
                pass  # Директория уже существует

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{timestamp}_{filename}"
            self.sftp.rename(f"{SFTP_REMOTE_PATH}/{filename}",
                             f"{archive_dir}/{new_name}")
            logger.info(f"Archived: {filename} -> {archive_dir}/{new_name}")
            return True
        except Exception as e:
            logger.error(f"Archive failed for {filename}: {e}")
            return False

    def process_new_files(self):
        if not self.connect():
            return

        try:
            new_files = self.get_new_audio_files()
            if not new_files:
                logger.info("No new audio files to process")
                return

            logger.info(f"Found {len(new_files)} new audio files")

            for filename in new_files:
                # Читаем метаданные из txt файла
                metadata = self.read_metadata_file(filename)
                if not metadata:
                    logger.warning(f"No metadata found for {filename}, skipping")
                    continue

                local_file = self.download_audio_file(filename)
                if local_file:
                    try:
                        from main import process_audio_file
                        if process_audio_file(local_file, metadata):
                            self.save_processed_file(filename)
                            self.archive_processed_file(filename)
                            os.remove(local_file)  # Удаляем локальную копию
                    except Exception as e:
                        logger.error(f"Processing failed for {filename}: {e}")
        finally:
            self.close()

    def close(self):
        if self.sftp:
            self.sftp.close()
        self.ssh.close()
        logger.info("SFTP connection closed")