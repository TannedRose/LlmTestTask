from flask import Flask, jsonify, render_template, request, json
from openai import OpenAI

import chromadb

import os


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

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-unsafe')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    query = data['message']

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
            answer = results['documents'][0][0]
        else:
            context = results['documents'][0][0]
            prompt = (f"Use the following context to answer the question. Context: {context}"
                      f" Question: {query} Answer:")
            response = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            answer = response.choices[0].message.content
    response = {'message': answer}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)