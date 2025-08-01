# Stage 1: Builder
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for some Python packages)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements.txt separately for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user \
    torch==2.3.0+cpu \
    torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install only runtime dependencies with fixed package names
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create unprivileged user
RUN useradd -m appuser

# Switch to appuser
USER appuser
WORKDIR /home/appuser/app

# Copy installed Python packages from builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p downloaded_audio logs

# Set environment variables
ENV PYTHONPATH=/home/appuser/.local/lib/python3.12/site-packages:$PYTHONPATH \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Startup command
CMD ["python", "scheduler.py"]