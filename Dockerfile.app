FROM python:3.11-slim

WORKDIR /app

COPY requirements.flask.txt .
RUN pip install --no-cache-dir -r requirements.flask.txt

COPY app/ ./app/

RUN mkdir -p /app/chromadb_faq_openai

WORKDIR /app/app

EXPOSE 5000

CMD ["python", "app.py"]