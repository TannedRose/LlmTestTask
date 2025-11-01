from collections import defaultdict

from flask import Flask, jsonify, render_template, request
from openai import OpenAI, OpenAIError

import chromadb
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не задан в переменных окружения")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

try:
    client = chromadb.PersistentClient(path="chromadb_faq_openai")
    collection = client.get_collection(name="faq_collection_openai")
except Exception as e:
    logger.error(f"Ошибка при инициализации ChromaDB: {e}")
    raise RuntimeError("Не удалось подключиться к коллекции ChromaDB") from e

app = Flask(__name__)
user_context = defaultdict(list)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-unsafe')


def resolve_entity_with_context(query, context):
    """Преобразует query в более ясный поисковый запрос, учитывая context."""
    if not isinstance(context, list):
        logger.warning(f"Context не является списком: {context}, type: {type(context)}. Используем пустой список.")
        context = []
    if not all(isinstance(item, str) for item in context):
         logger.warning(f"Context содержит не строковые элементы: {context}. Оставляем только строковые.")
         context = [item for item in context if isinstance(item, str)]

    if not isinstance(query, str):
        logger.warning(f"Query не является строкой: {query}, type: {type(query)}. Используем пустую строку.")
        query = ""

    context_str = "\n".join(context)
    resolution_prompt = (
        f"Текущий вопрос пользователя: '{query}'.\n"
        f"Предыдущие сообщения пользователя (контекст, порядок важен): \n{context_str}\n\n"
        "Вопрос '{query}' содержит местоимение (например, 'он', 'она', 'это'). "
        "Твоя задача - понять, ЧТО именно обозначает это местоимение, ССЫЛАЯСЬ НА ПОСЛЕДНИЕ СООБЩЕНИЯ в КОНТЕКСТЕ.\n\n"

        "ИНСТРУКЦИЯ:\n"
        "1. Проанализируй КОНТЕКСТ С КОНЦА (начиная с последнего сообщения) на предмет упоминаний конкретных сущностей (например, 'семинар по ...').\n"
        "2. Найди БЛИЖАЙШУЮ к текущему вопросу '{query}' сущность, КОТОРАЯ может быть объектом местоимения 'он/она/это'.\n"
        "3. Замени местоимение в вопросе '{query}' на НАЗВАНИЕ найденной сущности.\n"
        "4. Если подходящей сущности нет, верни оригинальный вопрос {query}.\n\n"

        "ПРИМЕР:\n"
        "Контекст: ['Когда будет семинар по налогам?', 'а будет семинар по wildberries?']\n"
        "Текущий вопрос: 'Где он будет проходить?'\n"
        "Анализ: Последнее упоминание сущности - 'семинар по wildberries?'. 'он' скорее всего про него.\n"
        "Результат: Где будет проходить семинар по wildberries?\n\n"

        "Финальный уточнённый поисковый запрос для вопроса '{query}':"
    )
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": resolution_prompt}],
            max_tokens=100,
            temperature=0.0
        )
        refined_query_for_search = response.choices[0].message.content.strip()
        if refined_query_for_search.startswith("Финальный уточнённый поисковый запрос для вопроса"):
             colon_index = refined_query_for_search.find(':')
             if colon_index != -1:
                 refined_query_for_search = refined_query_for_search[colon_index+1:].strip()
             else:
                 logger.warning(f"Модель вернула неожиданный формат: {refined_query_for_search}. Используем как есть.")
        logger.info(f"Уточнённый запрос для поиска: {refined_query_for_search}")
        return refined_query_for_search
    except OpenAIError as e:
        logger.error(f"Ошибка OpenAI при разрешении сущности/уточнении запроса: {e}")
        return query
    except Exception as e:
        logger.error(f"Неожиданная ошибка при разрешении сущности/уточнении запроса: {e}")
        return query

def get_embedding(text):
    try:
        response = client_openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except OpenAIError as e:
        logger.error(f"Ошибка OpenAI при создании эмбеддинга: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()
        query = data['message']
        user_id = str(data['user_id'])

        context = user_context[user_id][-3:]
        user_context[user_id].append(query)

        resolved_query_for_search = resolve_entity_with_context(query, context)

        try:
            embedding = get_embedding(resolved_query_for_search)
            query_embedding = [embedding]
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            return jsonify({"message": "Произошла ошибка, попробуйте позже..."})

        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=10
            )
        except Exception as e:
            logger.error(f"Ошибка при запросе к ChromaDB: {e}")
            return jsonify({"message": "Произошла ошибка, попробуйте позже..."})

        threshold = 1.4
        filtered_docs = []
        for doc, dist in zip(results['documents'][0], results['distances'][0]):
            if dist < threshold:
                filtered_docs.append(doc)

        if not filtered_docs:
            answer = "Не могу найти информацию по вашему вопросу о семинарах."
        else:
            raw_info = "\n\n".join(filtered_docs)
            prompt = (
                "Ты — профессиональный консультант Белагропромбанка. "
                "Ответь на вопрос пользователя, используя только предоставленную информацию о семинарах.\n\n"
                f"- Вопрос пользователя: «{query}»\n"
                f"- Предоставленная информация (результаты поиска по уточнённому запросу '{resolved_query_for_search}'):\n{raw_info}\n\n"
                f"- Контекст (предыдущие сообщения пользователя, для понимания стиля или уточнений, если необходимо): {list(reversed(context))}\n"
                "Сформулируй структурированный, дружелюбный и точный ответ на вопрос «{query}». "
                "Не добавляй ничего от себя (без приветствия)."
            )

            try:
                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700
                )
                answer = response.choices[0].message.content
            except OpenAIError as e:
                logger.error(f"Ошибка OpenAI при генерации ответа: {e}")
                return jsonify({"message": "Произошла ошибка, попробуйте позже..."})
            except Exception as e:
                logger.error(f"Неожиданная ошибка при вызове OpenAI: {e}")
                return jsonify({"message": "Произошла ошибка, попробуйте позже..."})

        response = {'message': answer}
        return jsonify(response)
    except Exception as e:
        logger.error(f"Необработанная ошибка в функции /send_message: {e}")
        return jsonify({"message": "Произошла ошибка, попробуйте позже..."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
