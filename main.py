import os
import traceback
from datetime import datetime

from asr import transcribe_audio
from nlu import load_llm_model, parse_voice_claim
from config import *
from glpi_api import connect
import logging

logger = logging.getLogger(__name__)

# Инициализация моделей при старте
tokenizer, model = None, None


def init_models():
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Initializing ML models...")
        tokenizer, model = load_llm_model(LLM_MODEL_NAME)


def generate_ticket_content(claim_data):
    """Генерация содержимого заявки"""
    return (
        "Заявка создана через голосового помощника\n"
        "#голосовой_помощник\n"
        f"Поезд: {claim_data.get('train_number', 'N/A')}\n"
        f"Вагон: {claim_data.get('wagon_number', 'N/A')}\n"
        f"Серийный номер: {claim_data.get('wagon_sn', 'N/A')}\n"
        f"Проблемы: {', '.join(claim_data.get('problems', []))}\n"
        f"Заявитель: {claim_data.get('executor_name', 'аноним')}\n"
        f"Номер звонящего: {claim_data.get('callerid', 'N/A')}\n"
        f"Дата звонка: {claim_data.get('call_date', 'N/A')}"
    )


def process_audio_file(audio_path, metadata=None):
    init_models()

    try:
        # Распознавание аудио
        recognized_text = transcribe_audio(audio_path, WHISPER_MODEL_PATH, DEVICE, COMPUTE_TYPE)
        if not recognized_text:
            logger.error("No text recognized from audio")
            return False

        # Извлечение структурированных данных
        claim_data = parse_voice_claim(recognized_text, tokenizer, model)
        logger.info(f"Extracted data: {claim_data}")

        # Добавляем метаданные из txt файла в claim_data
        if metadata:
            claim_data.update({
                'callerid': metadata.get('callerid', 'N/A'),
                'origtime': metadata.get('origtime', 0),
                'call_date': datetime.fromtimestamp(metadata.get('origtime', 0)).strftime('%Y-%m-%d %H:%M:%S')
            })

        # Создание заявки в GLPI
        with connect(GLPI_URL, GLPI_APP_TOKEN, GLPI_USER_TOKEN) as glpi:
            ticket_data = {
                "name": f"Заявка от {claim_data.get('executor_name', 'анонима')}",
                "content": generate_ticket_content(claim_data),
                "urgency": 4,
                "impact": 4,
                "priority": 4,
                "type": 1,
                "requesttypes_id": 8,
                "itilcategories_id": 39,
                "entities_id": 16,
                "_users_id_observer": [22],
            }

            result = glpi.add("Ticket", ticket_data)
            ticket_id = result[0]['id']
            logger.info(f"Ticket #{ticket_id} created successfully")
            return True

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        traceback.print_exc()
        return False

    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Deleted audio file: {audio_path}")
        except Exception as e:
            logger.error(f"Error deleting audio file {audio_path}: {e}")


if __name__ == "__main__":
    # Для ручного тестирования
    process_audio_file("msg0003.wav")
