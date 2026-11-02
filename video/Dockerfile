# Сервис кружочков Мики. Собирается из корня комплекта (/home/user/apps/mika на .90):
#   docker compose -f video/docker-compose.yml up -d --build
# Код, модели и страница аватара в образ не входят — каталог подключается томом только для чтения.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# libegl1 и libvulkan1 — загрузчики, без них Chrome не видит карту и рисует процессором (250 мс на кадр
# против 5 мс на 3090, проверено 31.10.2026). sox нужен qwen_tts при импорте.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg sox libegl1 libvulkan1 wget ca-certificates \
 && wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
 && apt-get install -y --no-install-recommends ./google-chrome-stable_current_amd64.deb \
 && rm google-chrome-stable_current_amd64.deb && rm -rf /var/lib/apt/lists/*

# numpy заранее: sox (зависимость qwen_tts) собирается из исходников и импортирует его прямо в setup.py
RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126 \
 && pip3 install --no-cache-dir numpy
COPY video/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    CHROME=/usr/bin/google-chrome FFMPEG=ffmpeg \
    CHROME_FLAGS="--use-angle=vulkan --enable-features=Vulkan --no-sandbox"
WORKDIR /mika/video
CMD ["python3", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8190"]
