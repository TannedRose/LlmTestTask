# Project: LlmTestTask

<p style="text-align: center;">
</p>
it's just a bot for answering FAQ based on openai and RAG

___


## Contents:
- [Description](#description)
- [Installation and starting](#installation-and-starting)

---

## Technologies


**Programming languages and modules:**

[![Python](https://img.shields.io/badge/-python_3.11^-464646?logo=python)](https://www.python.org/)


**Frameworks:**

[![Flask](https://img.shields.io/badge/-Flask-464646?logo=flask)](https://flask.palletsprojects.com/en/stable/)
[![Aiogram](https://img.shields.io/badge/-Aiogram-464646?logo=telegram)](https://aiogram.dev/)

**Containerization:**

[![docker](https://img.shields.io/badge/-Docker-464646?logo=docker)](https://www.docker.com/)
[![docker_compose](https://img.shields.io/badge/-Docker%20Compose-464646?logo=docker)](https://docs.docker.com/compose/)

[⬆️Contents](#contents)

---

## Installation and starting

<details><summary>Pre-conditions</summary>

It is assumed that the user has installed [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) on the local machine or on the server where the project will run. You can check if they are installed using the command:

```bash
docker --version && docker-compose --version
```
</details>


Local launch:

1. Clone the repository from GitHub:
```bash
https://github.com/TannedRose/LlmTestTask.git
```

1.1 Enter the data for the environment variables in the [.env] file:

```
OPENAI_API_KEY=
FLASK_SECRET_KEY=
BOT_TOKEN=

```


<details><summary>Lounch via Docker: Docker Compose</summary>

2. From the root directory of the project, execute the command:
```bash
docker-compose up --build
```

3. You can stop docker and delete containers with the command from the root directory of the project:
```bash
docker-compose down
```
add flag -v to delete volumes ```docker-compose down -v```
</details><h1></h1>

[⬆️Contents](#contents)