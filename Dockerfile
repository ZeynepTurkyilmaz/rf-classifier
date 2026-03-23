FROM python:3.11-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential cmake libfftw3-dev \
    python3-dev pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python paketleri
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# C++ modülünü derle
RUN cd cpp && \
    g++ -O2 -shared -fPIC \
    $(python3 -m pybind11 --includes) \
    fft_module.cpp \
    -o fft_module$(python3-config --extension-suffix) \
    -lfftw3 -lm

# .so dosyasını api ve model klasörlerinin bulabileceği yere taşı
RUN cp cpp/fft_module*.so /app/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
