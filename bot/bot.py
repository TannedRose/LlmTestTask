from aiogram import F
import asyncio
from aiogram.types import Message
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
import json
import requests

import os


bot_token = os.getenv("BOT_TOKEN")

url = 'http://flask:5000/'

bot = Bot(str(bot_token))
dp = Dispatcher()

# handling command /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello, I'm bot based on beluga llm, ready to answer your questions.")

@dp.message(F.text)
async def receive_prompt(message: Message):
    prompt = message.text
    jsonData={'message': prompt}
    jsonResponse = requests.post(
        url+'/send_message',
        json=jsonData
    )

    response = jsonResponse.json()
    await message.answer(response['message'])


# initialize polling to wait for incoming messages
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())