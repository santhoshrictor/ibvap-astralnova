FROM python:3.10-slim

# Install system dependencies for OpenCV, FFmpeg, and PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user according to Hugging Face Spaces best practices (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Ensure user owns /app and cache directories
RUN chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1

# Install CPU PyTorch and Python dependencies
COPY --chown=user:user requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and models
COPY --chown=user:user . /app/

# Hugging Face Spaces default port
EXPOSE 7860

# Launch FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
