# Stage 1: Builder
FROM python:3.12-slim-bookworm as builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    cmake \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Копируем requirements.txt отдельно для кэширования
COPY requirements.txt .

RUN pip install --no-cache-dir --user \
    torch==2.3.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Установка зависимостей
RUN pip install --no-cache-dir --user -r requirements.txt

# Копируем исходный код
COPY . .

# Stage 2: Final image
FROM python:3.12-slim-bookworm

WORKDIR /app

# Runtime зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя
RUN adduser --disabled-login --gecos '' appuser
USER appuser
WORKDIR /home/appuser/app

# Копируем только необходимое из builder
COPY --from=builder --chown=appuser:appuser /root/.local/lib/python3.12/site-packages /home/appuser/.local/lib/python3.12/site-packages
COPY --from=builder --chown=appuser:appuser /app /home/appuser/app

# Настройка окружения
ENV PYTHONPATH=/home/appuser/.local/lib/python3.12/site-packages:$PYTHONPATH \
    PATH=/home/appuser/.local/bin:$PATH

# Команда запуска (убедитесь, что scheduler.py в корне проекта)
CMD ["python", "scheduler.py"]