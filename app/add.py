import json
import os
from datetime import datetime
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Загружаем переменные окружения (например, OPENAI_API_KEY)
load_dotenv()

# Путь к вашему JSON-файлу
JSON_PATH = "json2.json"
CHROMA_PATH = "app/app/chroma_db"
COLLECTION_NAME = "events"

# Настройка эмбеддингов
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def normalize_date(date_str: str) -> str:
    """
    Преобразует строку вида '04 ноября 2025, 12:00' в ISO 8601.
    """
    months = {
        "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
        "мая": "05", "июня": "06", "июля": "07", "августа": "08",
        "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"
    }
    parts = date_str.split()
    day = parts[0]
    month = months[parts[1]]
    year = parts[2].replace(",", "")
    time = parts[3]
    return f"{year}-{month}-{day.zfill(2)}T{time}:00"


def load_and_transform_data(json_path: str) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for region_block in data:
        region = region_block["Регион"]
        for event in region_block["Мероприятия"]:
            # ID — уникальный, но безопасный (убираем пробелы/знаки)
            base_id = f"{region}_{event['Город']}_{event['Название'].split()[0]}_{event['Дата и время'].split()[0]}"
            event_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in base_id.lower())

            doc_text = (
                f"{event['Название']} состоится {event['Дата и время']}, "
                f"{event['Город']}, {event['Адрес']}. "
                f"Докладчик: {event['Докладчик']}. "
                f"Темы: {', '.join(event['Вопросы для рассмотрения'])}."
            )

            metadata = {
                "region": region,
                "city": event["Город"],
                "address": event["Адрес"],
                "date_iso": normalize_date(event["Дата и время"]),  # ISO 8601
                "speaker": event["Докладчик"],
                "registration_url": event["Ссылка на регистрацию"],
                # Опционально: сырое поле для фильтрации позже
                "raw_date": event["Дата и время"],
            }

            ids.append(event_id)
            documents.append(doc_text)
            metadatas.append(metadata)

    return documents, metadatas, ids


def ingest_to_chroma():
    documents, metadatas, ids = load_and_transform_data(JSON_PATH)

    # Инициализируем векторное хранилище (Chroma через LangChain)
    vectorstore = Chroma(
        collection_name="events",
        embedding_function=embedding_model,
        persist_directory="app/app/chroma_db",
    )

    # Добавляем документы
    vectorstore.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Загружено {len(documents)} мероприятий в коллекцию '{COLLECTION_NAME}'")
    try:
        count = vectorstore._collection.count()
        print(f"[DIAG] Кол-во в коллекции: {count}")
    except Exception as e:
        print(f"[DIAG] ОШИБКА при collection.count(): {type(e).__name__}: {e}")
        raise  # или return jsonify({"error": str(e)})
    return vectorstore


# Вызов при запуске напрямую
if __name__ == "__main__":
    import shutil
    if os.path.exists("app/app/chroma_db"):
        shutil.rmtree("app/app/chroma_db")
    ingest_to_chroma()