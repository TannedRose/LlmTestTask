from collections import defaultdict

from flask import Flask, jsonify, render_template, request, json
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
    raise RuntimeError("Не удалось подключиться к коллекции ChromaDB 'faq_collection_openai'") from e


app = Flask(__name__)
user_context = defaultdict(list)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-unsafe')


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
        user_context[user_id].append({'role': 'user', 'text': query})

        try:
            query_embedding = [get_embedding(query)]
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            return jsonify({"message" : "Произошла ошибка, попробуйте позже..."})

        try:
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=1
            )
        except Exception as e:
            logger.error(f"Ошибка при запросе к ChromaDB: {e}")
            return jsonify({"message" : "Произошла ошибка, попробуйте позже..."})


        if len(results['documents'][0]) == 0:
            answer = "Не могу найти ответ на этот вопрос."
        else:
            distances = results['distances'][0][0] if results['distances'] else float('inf')
            threshold = 0.9

            if distances < threshold:
                pre_answer = results['documents'][0][0]
                prompt = (
                    "Ты — профессиональный консультант белорусского банка «Паритетбанк». "
                    "Твоя задача — дать один чёткий, вежливый и точный ответ **только на последний вопрос пользователя**. "
                    "Не отвечай на предыдущие сообщения или вопросы из истории — они приведены лишь для понимания контекста "
                    "(например, чтобы расшифровать местоимения вроде «это», «там», «как там с оформлением?» и т.п.).\n\n"
    
                    "Информация для ответа:\n"
                    f"— Последний вопрос пользователя: «{query}»\n"
                    f"— Контекст переписки (только для понимания смысла): {context}\n"
                    f"— Сырой ответ из внутренней базы знаний: {pre_answer}\n\n"
    
                    "На основе этого:\n"
                    "1. Сформулируй **один** ответ строго на последний вопрос.\n"
                    "2. Используй данные из сырого ответа, но адаптируй их под формулировку вопроса и контекст.\n"
                    "3. Не упоминай, что используешь базу знаний, контекст или ИИ — отвечай как уверенный сотрудник банка.\n"
                    "4. Не добавляй информацию, которой нет в сырой базе знаний.\n"
                    "5. Ответ должен быть структурированным, понятным и соответствовать официальному, но дружелюбному тону банка."
                )

                try:
                    response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                    )
                    answer = response.choices[0].message.content
                except OpenAIError as e:
                    logger.error(f"Ошибка OpenAI при генерации ответа: {e}")
                    return jsonify({"message": "Произошла ошибка, попробуйте позже..."})
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при вызове OpenAI: {e}")
                    return jsonify({"message": "Произошла ошибка, попробуйте позже..."})
            else:
                answer = "Не могу найти ответ на этот вопрос."

        response = {'message': answer}
        return jsonify(response)
    except Exception as e:
        logger.error(f"Необработанная ошибка в функции /send_message: {e}")
        return jsonify({"message": "Произошла ошибка, попробуйте позже..."})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)