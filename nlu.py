# nlu.py – улучшенная версия с сохранёнными именами функций
import json
import re
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_llm_model(model_name: str):
    """load_llm_model – оставляем имя без изменений."""
    logger.info("🧠 Загружаем LLM: %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logger.info("✅ Модель загружена на %s", model.device)
    return tok, model


def extract_first_json(text: str):
    """extract_first_json – старое имя."""
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_data_from_text_fallback(text: str):
    """extract_data_from_text_fallback – старое имя."""
    data = {}

    # поезд
    train_match = re.search(r"(?:поезд|train)\s*(?:№|#)?\s*(\d+)", text, re.I)
    data["train_number"] = train_match.group(1) if train_match else None

    # вагон
    m = re.search(r"(?:вагон|wagon)\s*(?:№|#)?\s*(\d{1,2})", text, re.I)
    data["wagon_number"] = m.group(1) if m else None

    # серийник (по умолчанию None)
    data["wagon_sn"] = None

    # ФИО
    m = re.search(r"\b([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\b", text)
    data["executor_name"] = m.group(1) if m else None

    # проблемы
    probs = re.findall(
        r"(?:проблема|не работает|сломан|отключ|пропадает|не ид[её]т)\s*[а-яёa-z0-9,\s\-]{3,}",
        text, re.I
    )
    data["problems"] = [p.strip() for p in probs] or [text[:120].strip()]
    return data


def parse_voice_claim(text: str, tokenizer, model):
    """parse_voice_claim – старое имя."""
    prompt = (
        'Извлеки данные и верни только JSON: {"train_number":"", "wagon_number":"", '
        '"wagon_sn":"", "problems":[], "executor_name":""}\n\n'
        '- train_number – только цифры, без букв и слов '
        f'Текст: "{text}"\n'
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=120,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    response = tokenizer.decode(ids[0], skip_special_tokens=True)[len(prompt):].strip()

    result = extract_first_json(response)
    if result:
        data = {k: result.get(k) for k in ("train_number", "wagon_number", "wagon_sn", "problems", "executor_name")}
        data["problems"] = list(map(str, data.get("problems") or []))
        logger.info("✅ JSON из LLM получен")
    else:
        data = extract_data_from_text_fallback(text)
        logger.warning("⚠️ Используем fallback")

    return data