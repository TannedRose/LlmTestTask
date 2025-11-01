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
        context_str = "\n".join(context)
        logger.error(context)
        logger.error(context_str)
        user_context[user_id].append(query)

        try:
            embedding = get_embedding(query)
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
                "Ответь на последний вопрос пользователя, используя только предоставленные семинары.\n\n"
                f"- Вопрос: «{query}»\n"
                f"- Семинары:\n{raw_info}\n\n"
                f"- Контекст(для понимания местоимений): {context_str}"
                "Сформулируй структурированный, дружелюбный и точный ответ. Не добавляй ничего от себя(без приветствия)."
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
