import json
import logging
from collections import defaultdict

from flask import Flask, jsonify, render_template, request
from langchain.chat_models import ChatOpenAI
from vector_db import VectorDB
from langchain_chroma import Chroma
import config
import dateparser
from datetime import datetime, timedelta
import re

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Flask ---
app = Flask(__name__)
user_context = defaultdict(list)
app.config['SECRET_KEY'] = 'dev-secret-key-unsafe'

vector_db = VectorDB()
vector_db.vector_store = Chroma(
    persist_directory=vector_db.persist_directory,
    embedding_function=vector_db.embedding_model,
    collection_name="events_collection",
)
collection = vector_db.vector_store

chat_client = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=config.OPENAI_API_KEY)

def parse_date_range(text):
    now = datetime.now().date()

    match = re.search(r'(\d+)-е числа', text)
    if match:
        day = int(match.group(1))
        start = now.replace(day=day)
        end = now.replace(day=min(day+9,28))
        return start, end

    match = re.search(r'после\s+(\d{1,2}\s[а-я]+)', text.lower())
    if match:
        start = dateparser.parse(match.group(1), languages=["ru"])
        if start:
            return start.date(), None

    parsed = dateparser.parse(text, languages=["ru"])
    if parsed:
        return parsed.date(), parsed.date()

    return None, None

def resolve_entity_with_context(query, context, max_context=5):

    context_slice = context[-max_context:]
    context_str = "\n".join(str(c) for c in context_slice)
    prompt = (
        f"Вопрос: {query}\nКонтекст (последние сообщения):\n{context_str}\n"
        "Если есть местоимения, замени их на конкретные сущности. "
        "Ответ только итоговым вопросом."
    )
    try:
        response = chat_client.invoke([{"role": "user", "content": prompt}])
        return response.content.strip() or query
    except Exception as e:
        logger.error(f"Ошибка при уточнении запроса: {e}")
        return query

def filter_events_by_date_and_speaker(events, start_date=None, end_date=None, speaker=None):
    filtered = []

    for event_text in events:
        if isinstance(event_text, dict):
            # Если вдруг пришёл dict — берём текст из ключа
            event_text = event_text.get("text", str(event_text))

        date_match = re.search(r'\d{1,2}\s[а-яё]+ \d{4}', event_text, re.IGNORECASE)
        event_date = None
        if date_match:
            try:
                event_date = dateparser.parse(date_match.group(0), languages=["ru"]).date()
            except Exception:
                pass

        # Проверка по дате
        if start_date and event_date and event_date < start_date:
            continue
        if end_date and event_date and event_date > end_date:
            continue

        if speaker and speaker.lower() not in event_text.lower():
            continue

        filtered.append(event_text)

    return filtered



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()
        query = data.get('message', '')
        user_id = str(data.get('user_id', 'anon'))

        context = user_context[user_id][-3:]
        user_context[user_id].append(query)

        resolved_query = resolve_entity_with_context(query, context)
        logger.info(f"уточненный вопрос {resolved_query}")
        results = collection.similarity_search_with_score(query, k=10)
        threshold = 1.4
        events = []
        logger.info(f"EVENTS: {events}")
        for doc, score in results:
            if score < threshold:
                content = doc.page_content.strip()
                if not content:
                    logger.warning("Пустое событие пропущено")
                    continue
                try:
                    event = json.loads(content)
                except json.JSONDecodeError:
                    event = {"Описание": content}
                events.append(event)

        start_date, end_date = parse_date_range(query)
        speaker_match = re.search(r'с\s+([А-ЯЁа-яё\s-]+)', query)
        speaker = speaker_match.group(1).strip() if speaker_match else None

        filtered_events = filter_events_by_date_and_speaker(events, start_date, end_date, speaker)

        if not filtered_events:
            answer = "Информация по вашему запросу не найдена."
        else:
            raw_info = "\n\n".join(filtered_events)
            prompt = f"""
Ты — ассистент по мероприятиям. Используй только данные из базы.
Информация из базы:
{raw_info}

Вопрос пользователя: {query}

Инструкции:
1. Используй только данные из базы, не придумывай.
2. Выделяй ключевые сущности: Дата и время, Адрес/место проведения, Спикер/докладчик, Название мероприятия.
3. Если вопрос про конкретное имя (спикера), ищи только совпадения по полному имени.
4. Если ничего не найдено — честно скажи, что информации нет.
5. Формат ответа: кратко, дружелюбно, точные факты. Каждое мероприятие отдельным пунктом.
Ответь только итоговой информацией.
"""
            try:
                response = chat_client.invoke([{"role": "user", "content": prompt}])
                answer = response.content.strip()
            except Exception as e:
                logger.error(f"Ошибка при генерации ответа: {e}")
                answer = "Произошла ошибка при генерации ответа."

        return jsonify({"message": answer})

    except Exception as e:
        logger.error(f"Необработанная ошибка в /send_message: {e}")
        return jsonify({"message": "Произошла ошибка, попробуйте позже..."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
