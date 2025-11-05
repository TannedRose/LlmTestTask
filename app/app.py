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
    chroma_client = chromadb.PersistentClient(path="chromadb_faq_openai")
    collection = chroma_client.get_collection(name="faq_collection_openai")
except Exception as e:
    logger.error(f"Ошибка при инициализации ChromaDB: {e}")
    raise RuntimeError("Не удалось подключиться к коллекции ChromaDB") from e

app = Flask(__name__)
user_context = defaultdict(list)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-unsafe')


def resolve_entity_with_context(query, context):
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
        f"Вопрос: {query}\n"
        f"Контекст (последние сообщения):\n{context_str}\n\n"
        "Если в вопросе есть местоимение (он/она/оно/это/тот/та и т.п.), "
        "замени его на конкретную сущность из контекста. "
        "Анализируй контекст снизу вверх и выбирай ближайшее подходящее упоминание.\n"
        "Если подходящей сущности нет — оставь вопрос без изменений.\n"
        "Ответ должен содержать ТОЛЬКО итоговый вопрос. Ничего больше не пиши."
    )
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": resolution_prompt}],
            max_tokens=100
        )
        refined_query_for_search = response.choices[0].message.content.strip()
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
                n_results=7
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
                "Ты — профессиональный консультант ОАО «Белагропромбанк». "
                "Твоя задача — ответить на вопрос пользователя, используя ТОЛЬКО предоставленную информацию о семинарах.\n\n"

                "СТРОГИЕ ПРАВИЛА:\n"
                "- Отвечай ЧИСТЫМ ТЕКСТОМ. НИКАКИХ символов форматирования!\n"
                "- ЗАПРЕЩЕНО использовать: *, **, __, ##, ``` , <b>, <i>, [текст](ссылка), **любые звёздочки и угловые скобки**.\n"
                "- Ссылки пиши как обычный URL (например: https://example.com).\n"
                "- Не выделяй заголовки жирным. Вместо '**Дата:**' пиши просто 'Дата:'.\n"
                "- Используй только буквы, цифры, пробелы, дефисы, точки, двоеточия, переносы строк.\n\n"

                f"Вопрос пользователя: «{query}»\n"
                f"Уточнённый поисковый запрос: «{resolved_query_for_search}»\n"
                f"Релевантная информация из базы знаний:\n{raw_info}\n"
                f"Контекст диалога (предыдущие сообщения): {list(reversed(context))}\n\n"

                "Сформулируй краткий, дружелюбный и точный ответ. "
                "Не добавляй приветствий, прощаний или фраз вроде 'Вот информация'. "
                "Начни сразу с сути. Ответ должен быть готов к отправке в Telegram без ошибок."
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

        return jsonify({"message": answer})
    except Exception as e:
        logger.error(f"Необработанная ошибка в функции /send_message: {e}")
        return jsonify({"message": "Произошла ошибка, попробуйте позже..."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
