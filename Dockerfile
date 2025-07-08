# Используем официальный образ NVIDIA CUDA в качестве базового
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Устанавливаем переменные окружения, чтобы избежать интерактивных запросов при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем системные зависимости, включая Python и ffmpeg
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем псевдонимы для python и pip
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Копируем остальной код приложения
COPY . .

# Указываем команду для запуска приложения
CMD ["python", "scheduler.py"]