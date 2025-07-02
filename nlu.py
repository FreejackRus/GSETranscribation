import logging
import json
import regex as re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Настройка логгирования
logger = logging.getLogger(__name__)


def load_llm_model(model_name: str):
    """
    Загружает языковую модель и токенизатор.
    """
    logger.info(f"🧠 Загружаем LLM: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"✅ Модель загружена на устройство: {model.device}")
        return tokenizer, model
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке модели: {e}")
        raise

def extract_first_json(text: str) -> dict | None:
    """
    Ищет первый попавшийся JSON или JSON-like объект в строке и возвращает его как dict.
    Поддерживает вложенные структуры благодаря рекурсивному регулярному выражению.
    """
    logger.debug("🔍 Поиск JSON в ответе модели...")

    # Регулярное выражение для поиска JSON-объектов и массивов
    pattern = r'\{(?:[^{}]|(?R))*\}|\$$(?:[^\$$]|(?R))*\$$'

    try:
        matches = re.findall(pattern, text, flags=re.DOTALL)
    except Exception as e:
        logger.error(f"❌ Ошибка компиляции регулярного выражения: {e}")
        return None

    for i, match in enumerate(matches):
        logger.debug(f"🔎 Найден возможный JSON (вариант {i + 1}): {match[:200]}...")

        try:
            result = json.loads(match)
            logger.info("✅ JSON успешно распарсен")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"❌ Ошибка парсинга JSON (вариант {i + 1}): {e}")

    logger.error("❌ JSON не найден или некорректен")
    return None


def parse_voice_claim(text: str, tokenizer, model) -> dict:
    """
    Парсит голосовое обращение через LLM и возвращает структурированные данные.
    Если LLM не вернул JSON — использует fallback-парсер на регулярках.
    """

    prompt = f"""
    ВАЖНО: Верни только JSON. Не повторяй промпт. Не добавляй текст. Работай только с текстом: "{text}".

    Вы — помощник технической поддержки. Ваша задача — извлечь данные из голосового обращения пользователя.

    Проанализируйте текст и верните только **валидный JSON-объект** со следующими полями:
    - train_number (строка (номер поезда))
    - wagon_number (строка (номер вагона))
    - wagon_sn (строка (серийный номер вагона))
    - problems (массив строк (проблемы))
    - executor_name (ФИО)
    """

    logger.info("📝 Промпт, отправляемый в модель:")
    logger.info(prompt)

    logger.info("🧠 Генерируем ответ через LLM...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logger.info("🟢 Ответ модели получен:")
    logger.info(response)

    logger.info("🔍 Попытка извлечь JSON из ответа модели...")
    result = extract_first_json(response)

    if result:
        logger.info("✅ JSON успешно извлечён из ответа модели")
        logger.debug(f"📊 Извлечённые данные: {result}")

        return {
            "train_number": result.get("train_number"),
            "wagon_number": result.get("wagon_number"),
            "wagon_sn": result.get("wagon_sn"),
            "problems": result.get("problems", []),
            "executor_name": result.get("executor_name")
        }
    else:
        logger.warning("⚠️ JSON не найден в ответе модели, перехожу к fallback-парсеру")
        fallback_data = extract_data_from_text_fallback(text)
        logger.info("🔄 Fallback-данные:")
        logger.debug(fallback_data)
        return fallback_data


def extract_data_from_text_fallback(text: str) -> dict:
    """
    Альтернативный метод извлечения данных на основе регулярных выражений.
    """
    data = {}

    # Поезд
    train_match = re.search(r"(?:поезд|номер\s+поезда).*?(\d+)", text, re.IGNORECASE)
    data["train_number"] = train_match.group(1) if train_match else None

    # Вагон
    wagon_match = re.search(r"(?:вагон|номер\s+вагона).*?(\d{1,2})", text, re.IGNORECASE)
    if wagon_match:
        wagon_num = wagon_match.group(1)
        if 1 <= int(wagon_num) <= 26:
            data["wagon_number"] = wagon_match.group(1)

    # Серийный номер вагона (не указан → None)
    data["wagon_sn"] = None

    # ФИО заявителя
    name_match = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)", text)
    if not name_match:
        name_match = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)", text)
    data["executor_name"] = name_match.group(1) if name_match else None

    # Проблемы
    problem_keywords = ["проблема", "не работает", "сломан", "нет", "не идет"]
    problem_parts = []

    for keyword in problem_keywords:
        parts = re.split(keyword, text, flags=re.IGNORECASE)
        if len(parts) > 1:
            problem_parts.append(parts[-1].strip())

    if problem_parts:
        problems = [p.strip() for p in " ".join(problem_parts).split(".") if p.strip()]
        data["problems"] = problems
    else:
        data["problems"] = [text[:100]]

    return data