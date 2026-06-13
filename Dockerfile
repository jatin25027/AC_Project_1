# ── HuggingFace Spaces Dockerfile ────────────────────────────────────────────
FROM python:3.11-slim

# System deps for matplotlib / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# HuggingFace Spaces runs as a non-root user — make results dir writable
RUN mkdir -p results/plots && chmod -R 777 results/

# HuggingFace Spaces default port
EXPOSE 7860

CMD ["python3", "backend_main.py"]
