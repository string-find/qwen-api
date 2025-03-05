FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies including build tools needed for some Python packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies with error verbosity
COPY requirements.txt .

# Install pip dependencies in separate steps for better debugging
RUN pip install --no-cache-dir --upgrade pip && \
    # Install qwen-vl-utils first to ensure it's properly installed
    pip install --no-cache-dir qwen-vl-utils==0.0.10 && \
    # Install the rest of the requirements
    pip install --no-cache-dir -r requirements.txt -v || cat /root/.cache/pip/log/debug.log

# Install flash-attn separately with proper build flags
RUN pip install --no-cache-dir "flash-attn" --no-build-isolation -v || cat /root/.cache/pip/log/debug.log

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
