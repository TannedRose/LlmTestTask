from aiogram import F
import asyncio
from aiogram.types import Message
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
import requests

import os

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


bot_token = os.getenv("BOT_TOKEN")
if not bot_token:
    raise ValueError("BOT_TOKEN не задан в переменных окружения")

url = 'http://flask:5000/'

bot = Bot(str(bot_token))
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я чат-бот помощник Паритет Банка. Отправь мне свой вопрос — и я постараюсь дать тебе качественный ответ!")


@dp.message(F.text)
async def receive_prompt(message: Message):
    prompt = message.text
    user_id = message.from_user.id

    try:
        jsonData={"user_id": user_id, 'message': prompt}

        jsonResponse = requests.post(
            url+'/send_message',
            json=jsonData,
        )

        response = jsonResponse.json()
        if 'message' in response:
            answer = response['message']
        else:
            answer = "Произошла ошибка, попробуйте позже..."
    except requests.exceptions.JSONDecodeError:
        logger.error("Ответ от сервера не является валидным JSON")
        answer = "Произошла ошибка, попробуйте позже..."
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP ошибка от Flask-сервера: {e} (статус: {jsonResponse.status_code})")
        answer = "Произошла ошибка, попробуйте позже..."
    except Exception as e:
        logger.error(f"Необработанная ошибка в функции receive_prompt: {e}")
        answer = "Произошла ошибка, попробуйте позже..."
    await message.answer(answer)


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())