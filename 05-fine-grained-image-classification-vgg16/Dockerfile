FROM python:3.12-slim

WORKDIR /app
COPY requirements-huggingface.txt ./
RUN pip install --no-cache-dir -r requirements-huggingface.txt

COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py"]
