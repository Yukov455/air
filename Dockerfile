FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Создаём директории для данных
RUN mkdir -p /app/data /app/models /app/logs

# Переменные окружения по умолчанию
ENV PORT=8080
ENV DATA_UPDATE_INTERVAL=300
ENV MODEL_TRAIN_INTERVAL=3600
ENV AUTO_START_SCHEDULER=true
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose порт
EXPOSE ${PORT}

# Запуск приложения
CMD ["python", "app.py"]
