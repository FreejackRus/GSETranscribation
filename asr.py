import logging

from pydub import AudioSegment
from faster_whisper import WhisperModel
from logger import setup_logger
import os

setup_logger()
logger = logging.getLogger(__name__)

def convert_to_16k_mono(input_path: str, output_path: str = None) -> str:
    if not os.path.exists(input_path):
        logger.error(f"Файл {input_path} не найден")
        raise FileNotFoundError(f"Файл {input_path} не найден")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted.wav"

    logger.info(f"Конвертируем {input_path} в 16 кГц, моно...")
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    logger.info(f"Конвертация завершена: {output_path}")
    return output_path


def transcribe_audio(file_path: str, model_path: str, device: str = "cpu", compute_type: str = "float16") -> str:
    converted_file = convert_to_16k_mono(file_path)

    logger.info("Загружаем модель Whisper...")
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    segments, info = model.transcribe(converted_file, beam_size=5, language="ru")
    logger.info(f"Обнаружен язык: {info.language}, уверенность: {info.language_probability:.2f}")

    full_text = " ".join(segment.text for segment in segments).strip()
    logger.info(f"Распознанный текст: {full_text}")
    return full_text