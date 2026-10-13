# Use official lightweight Python runtime
FROM python:3.11-slim

# Set environment variables for optimized execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set working directory inside container
WORKDIR /app

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK stopwords during build to avoid runtime delays
RUN python -m nltk.downloader stopwords

# Copy application source code and pre-trained artifacts
COPY . .

# Train model if .pkl files are not already present
RUN if [ ! -f spam_model.pkl ] || [ ! -f vectorizer.pkl ]; then python train_model.py; fi

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to monitor Streamlit container availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
