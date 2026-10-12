FROM python:3.13-slim

WORKDIR /app

COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

RUN ruff check . && ruff format --check .

CMD ["python", "examples/run_demo.py"]
