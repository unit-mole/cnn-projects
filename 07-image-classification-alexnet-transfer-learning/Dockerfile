FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "scripts/run_local_web_server.py", "--host", "0.0.0.0", "--port", "8000"]
