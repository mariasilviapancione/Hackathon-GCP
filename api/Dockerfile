FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-root


EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

