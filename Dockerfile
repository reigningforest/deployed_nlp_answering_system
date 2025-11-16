# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Configure spaCy cache directory.
# The app downloads en_core_web_md at runtime into this path if it is missing.
# In production, override SPACY_MODEL_DIR to a mounted persistent volume so the
# model is fetched only once (Railway: mount /data and set SPACY_MODEL_DIR=/data/spacy).
ENV SPACY_MODEL_DIR=/app/runtime_models/spacy

# Copy application code
COPY . .

# Pre-download spaCy model during build directly to the configured storage dir
RUN mkdir -p /app/runtime_models/spacy && \
    python -c "from src.rag.spacy_model import ensure_spacy_model; \
    ensure_spacy_model('en_core_web_md', '3.7.0', '/app/runtime_models/spacy')"

# Expose port (Railway/Render will use $PORT env var)
EXPOSE 8000

# Run the FastAPI app with uvicorn
# Wrap in sh -c so ${PORT:-8000} is interpreted at runtime.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
