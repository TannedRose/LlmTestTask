from collections import defaultdict

from flask import Flask, jsonify, render_template, request, json
from openai import OpenAI

import chromadb

import os

# import logging
#
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s'
# )
# logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не задан в переменных окружения")


client_openai = OpenAI(api_key=OPENAI_API_KEY)

client = chromadb.PersistentClient(path="chromadb_faq_openai")
try:
    collection = client.get_collection(name="faq_collection_openai")
except ValueError:
    raise RuntimeError("Коллекция 'faq_collection_openai' не найдена в ChromaDB")

def get_embedding(text):
    response = client_openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

app = Flask(__name__)
user_context = defaultdict(list)

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-unsafe')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    query = data['message']
    user_id = str(data['user_id'])

    context = user_context[user_id][-3:]
    user_context[user_id].append({'role': 'user', 'text': query})

    # logger.info(f"Последние сообщения от пользователя {user_id}:")
    # for msg in context:
    #     logger.info(f"  → {msg['role']}: {msg['text']}")

    query_embedding = [get_embedding(query)]
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )

    if len(results['documents'][0]) == 0:
        answer = "Не могу найти ответ на этот вопрос."
    else:
        distances = results['distances'][0][0] if results['distances'] else float('inf')
        threshold = 0.9

        if distances < threshold:
            pre_answer = results['documents'][0][0]
            prompt = (f"Ты являешься профессиональным консультантом в белорусском банке Paritet."
              f"Мне нужно получать максималоьно точный и структурированный ответ в коце."
              f"Используй Context и Answer (сырой ответ из базы знаний) чтобы вносить правки. "
              f"Никогда не упоминайте о контексте и о том, откуда взята информация."
              f"Context : {context}, Answer: {pre_answer}")
            response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
            )
            answer = response.choices[0].message.content
        else:
            answer = "Не могу найти ответ на этот вопрос."

    response = {'message': answer}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)