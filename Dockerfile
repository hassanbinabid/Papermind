# Hugging Face Spaces Docker setup for PaperMind RAG
FROM python:3.11-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install CPU-only PyTorch first to avoid downloading 532MB CUDA version
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user app/ ./app/
COPY --chown=user prompts/ ./prompts/
COPY --chown=user eval/ ./eval/
COPY --chown=user main.py .
COPY --chown=user chat.py .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Spaces requires port 7860
EXPOSE 7860

# Start command — note port 7860 for HF Spaces
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]